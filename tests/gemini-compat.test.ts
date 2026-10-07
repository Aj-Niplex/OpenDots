import { expect, it, vi } from 'vitest';
import {
  geminiToolCallFetch,
  isGoogleOpenAIEndpoint,
  restoreThoughtSignatures,
} from '../src/server/gemini-compat.js';

const PLACEHOLDER = 'skip_thought_signature_validator';
const ENDPOINT =
  'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions';

function sse(deltas: unknown[]) {
  const text =
    deltas
      .map(
        (delta) =>
          `data: ${JSON.stringify({
            id: 'completion',
            object: 'chat.completion.chunk',
            created: 1,
            model: 'gemini-3',
            choices: [{ index: 0, delta, finish_reason: null }],
          })}\n\n`,
      )
      .join('') + 'data: [DONE]\n\n';
  return new Response(text, {
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

function toolCall(id: string, signature: string, args = '{}') {
  return {
    id,
    type: 'function',
    function: { name: 'lookup', arguments: args },
    extra_content: { google: { thought_signature: signature } },
  };
}

function history(id: string, args = '{}') {
  return {
    model: 'gemini-3',
    messages: [
      { role: 'user', content: 'Look it up.' },
      {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id,
            type: 'function',
            function: { name: 'lookup', arguments: args },
          },
        ],
      },
      { role: 'tool', tool_call_id: id, content: '{}' },
    ],
  };
}

function signatureSent(init: RequestInit | undefined) {
  const body = JSON.parse(String(init?.body));
  return body.messages[1].tool_calls[0].extra_content?.google
    ?.thought_signature;
}

/** Passes a model response through the wrapper, as the OpenAI client would. */
async function receive(scope: string, response: Response) {
  const wrapped = geminiToolCallFetch(scope, async () => response);
  const result = await wrapped(ENDPOINT, {
    method: 'POST',
    body: JSON.stringify({ model: 'gemini-3', messages: [] }),
  });
  return result.text();
}

/** The signature the wrapper sends for a replayed tool call. */
async function replay(scope: string, body: unknown) {
  const inner = vi.fn<typeof fetch>().mockImplementation(async () => {
    return new Response('{}');
  });
  await geminiToolCallFetch(scope, inner)(ENDPOINT, {
    method: 'POST',
    body: JSON.stringify(body),
  });
  return signatureSent(inner.mock.calls[0][1]);
}

/** The response is scanned in the background, so wait for it to be learned. */
function eventually(scope: string, body: unknown, expected: string) {
  return vi.waitFor(async () => {
    expect(await replay(scope, body)).toBe(expected);
  });
}

it('recognizes only Google endpoints', () => {
  expect(
    isGoogleOpenAIEndpoint(
      'https://generativelanguage.googleapis.com/v1beta/openai/',
    ),
  ).toBe(true);
  expect(isGoogleOpenAIEndpoint('https://api.openai.com/v1')).toBe(false);
  expect(
    isGoogleOpenAIEndpoint(
      'https://generativelanguage.googleapis.com.evil.test/v1',
    ),
  ).toBe(false);
  expect(isGoogleOpenAIEndpoint('not a url')).toBe(false);
  expect(isGoogleOpenAIEndpoint(undefined)).toBe(false);
});

it('sends back the signature Gemini returned, and leaves the response readable', async () => {
  const text = await receive(
    'round-trip',
    sse([
      { role: 'assistant', tool_calls: [toolCall('call-real', 'SIG-REAL')] },
    ]),
  );
  expect(text).toContain('SIG-REAL');
  await eventually('round-trip', history('call-real'), 'SIG-REAL');
});

it('uses the documented placeholder for tool calls it has not seen', async () => {
  expect(await replay('unseen', history('call-from-before-restart'))).toBe(
    PLACEHOLDER,
  );
});

it('keeps a signature that is already present', () => {
  const body = history('call-own');
  Object.assign(body.messages[1].tool_calls![0], {
    extra_content: { google: { thought_signature: 'MINE' } },
  });
  expect(restoreThoughtSignatures(body)).toBe(0);
  expect(JSON.stringify(body)).toContain('MINE');
  expect(JSON.stringify(body)).not.toContain(PLACEHOLDER);
});

it('does not share signatures between conversations that reuse a tool-call ID', async () => {
  await receive(
    'conversation-a',
    sse([{ tool_calls: [toolCall('call_0', 'SIG-A')] }]),
  );
  await receive(
    'conversation-b',
    sse([{ tool_calls: [toolCall('call_0', 'SIG-B')] }]),
  );
  await eventually('conversation-a', history('call_0'), 'SIG-A');
  await eventually('conversation-b', history('call_0'), 'SIG-B');
  // A conversation that never saw the call gets the placeholder, not a stranger's.
  expect(await replay('conversation-c', history('call_0'))).toBe(PLACEHOLDER);
});

it('keeps the original signature when an ID is reused with other arguments', async () => {
  await receive(
    'reused-id',
    sse([{ tool_calls: [toolCall('call_0', 'SIG-1', '{"q":1}')] }]),
  );
  await receive(
    'reused-id',
    sse([{ tool_calls: [toolCall('call_0', 'SIG-2', '{"q":2}')] }]),
  );
  await eventually('reused-id', history('call_0', '{"q":1}'), 'SIG-1');
  await eventually('reused-id', history('call_0', '{"q": 2}'), 'SIG-2');
  // Ambiguous: two calls share the ID and neither matches, so do not guess.
  expect(await replay('reused-id', history('call_0', '{"q":3}'))).toBe(
    PLACEHOLDER,
  );
});

it('assembles fragmented stream deltas before associating the signature', async () => {
  await receive(
    'fragmented',
    sse([
      {
        role: 'assistant',
        tool_calls: [
          {
            index: 0,
            id: 'call-frag',
            type: 'function',
            function: { name: 'lookup', arguments: '' },
          },
        ],
      },
      { tool_calls: [{ index: 0, function: { arguments: '{"q":' } }] },
      {
        tool_calls: [
          {
            index: 0,
            function: { arguments: '1}' },
            extra_content: { google: { thought_signature: 'SIG-FRAG' } },
          },
        ],
      },
    ]),
  );
  await eventually('fragmented', history('call-frag', '{"q":1}'), 'SIG-FRAG');
});

it('keeps parallel streamed tool calls apart by index', async () => {
  await receive(
    'parallel',
    sse([
      {
        role: 'assistant',
        tool_calls: [
          {
            index: 0,
            id: 'call-a',
            type: 'function',
            function: { name: 'lookup', arguments: '{}' },
          },
          {
            index: 1,
            id: 'call-b',
            type: 'function',
            function: { name: 'lookup', arguments: '{}' },
          },
        ],
      },
      {
        tool_calls: [
          {
            index: 1,
            extra_content: { google: { thought_signature: 'SIG-B' } },
          },
        ],
      },
      {
        tool_calls: [
          {
            index: 0,
            extra_content: { google: { thought_signature: 'SIG-A' } },
          },
        ],
      },
    ]),
  );
  await eventually('parallel', history('call-a'), 'SIG-A');
  await eventually('parallel', history('call-b'), 'SIG-B');
});

it('reads signatures from a non-streaming completion', async () => {
  await receive(
    'json',
    new Response(
      JSON.stringify({
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [toolCall('call-json', 'SIG-JSON')],
            },
          },
        ],
      }),
      { headers: { 'Content-Type': 'application/json' } },
    ),
  );
  await eventually('json', history('call-json'), 'SIG-JSON');
});

it('does not touch other requests or non-JSON bodies', async () => {
  const inner = vi.fn<typeof fetch>().mockResolvedValue(new Response('{}'));
  const wrapped = geminiToolCallFetch('untouched', inner);
  const other = JSON.stringify(history('call-other'));
  await wrapped(
    'https://generativelanguage.googleapis.com/v1beta/openai/models',
    {
      method: 'POST',
      body: other,
    },
  );
  await wrapped(ENDPOINT, { method: 'POST', body: 'not json' });
  await wrapped(ENDPOINT, { method: 'GET' });
  expect(inner.mock.calls[0][1]?.body).toBe(other);
  expect(inner.mock.calls[1][1]?.body).toBe('not json');
  expect(inner.mock.calls[2][1]?.body).toBeUndefined();
});
