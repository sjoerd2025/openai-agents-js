import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { z } from 'zod';
import {
  Agent,
  FunctionTool,
  Handoff,
  ModelRequest,
  ToolNameCollisionPolicy,
  Usage,
  UserError,
  handoff,
  run,
  setDefaultModelProvider,
  setTracingDisabled,
  tool,
  toolNamespace,
} from '../../src';
import logger from '../../src/logger';
import {
  FakeModel,
  TEST_MODEL_FUNCTION_CALL,
  TEST_MODEL_RESPONSE_BASIC,
  FakeModelProvider,
} from '../stubs';

class RecordingModel extends FakeModel {
  readonly requests: ModelRequest[] = [];

  override async getResponse(request: ModelRequest) {
    this.requests.push(request);
    return super.getResponse(request);
  }
}

function functionTool(
  name: string,
  isEnabled: boolean = true,
  description: string = `Tool ${name}`,
  execute: () => string | Promise<string> = async () => name,
): FunctionTool<any, any, any> {
  return tool({
    name,
    description,
    parameters: z.object({}),
    isEnabled,
    execute,
  });
}

async function expectRejectedBeforeModelRequest(args: {
  tools?: FunctionTool<any, any, any>[];
  handoffs?: Handoff<any, any>[];
  expectedMessage: string;
  policy?: ToolNameCollisionPolicy;
}): Promise<void> {
  const model = new RecordingModel([TEST_MODEL_RESPONSE_BASIC]);
  const agent = new Agent({
    name: 'Collision agent',
    model,
    tools: args.tools,
    handoffs: args.handoffs,
  });

  await expect(
    run(agent, 'hello', {
      toolNameCollisionPolicy: args.policy ?? 'error',
    }),
  ).rejects.toMatchObject({
    name: UserError.name,
    message: args.expectedMessage,
  });
  expect(model.requests).toHaveLength(0);
}

describe('model-visible tool name validation', () => {
  setTracingDisabled(true);
  setDefaultModelProvider(new FakeModelProvider());

  beforeEach(() => {
    vi.spyOn(logger, 'dontLogToolData', 'get').mockReturnValue(false);
    vi.spyOn(logger, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const remediation =
    'Function tools and handoffs must have unique routed names. Assign unique tool names or toolNameOverride values, or use a namespace.';

  it('rejects duplicate enabled function tools before a model request', async () => {
    await expectRejectedBeforeModelRequest({
      tools: [functionTool('duplicate'), functionTool('duplicate')],
      expectedMessage: `Duplicate enabled tool name found: 'duplicate' (2 function tools). ${remediation}`,
    });
  });

  it('rejects duplicate enabled function tools within the same namespace', async () => {
    const duplicateTools = toolNamespace({
      name: 'crm',
      description: 'CRM tools',
      tools: [functionTool('duplicate'), functionTool('duplicate')],
    });

    await expectRejectedBeforeModelRequest({
      tools: [...duplicateTools],
      policy: 'warn',
      expectedMessage: `Duplicate enabled tool name found: 'crm.duplicate' (2 function tools). ${remediation}`,
    });
  });

  it.each<ToolNameCollisionPolicy>(['warn', 'error'])(
    'rejects duplicate deferred top-level function tools under the %s policy',
    async (policy) => {
      const first = functionTool('duplicate');
      const second = functionTool('duplicate');
      first.deferLoading = true;
      second.deferLoading = true;

      await expectRejectedBeforeModelRequest({
        tools: [first, second],
        policy,
        expectedMessage: `Duplicate enabled tool name found: 'duplicate' (2 function tools). ${remediation}`,
      });
    },
  );

  it('keeps a deferred top-level tool distinct from an equal bare name', async () => {
    const deferred = functionTool('lookup', true, 'Deferred tool');
    deferred.deferLoading = true;
    const model = new RecordingModel([TEST_MODEL_RESPONSE_BASIC]);
    const agent = new Agent({
      name: 'Deferred identity agent',
      model,
      tools: [functionTool('lookup', true, 'Bare tool'), deferred],
    });

    await run(agent, 'hello');

    expect(model.requests[0]?.tools).toHaveLength(2);
    expect(logger.warn).not.toHaveBeenCalled();
  });

  it('rejects duplicate enabled handoffs before a model request', async () => {
    const first = handoff(new Agent({ name: 'First target' }), {
      toolNameOverride: 'duplicate',
    });
    const second = handoff(new Agent({ name: 'Second target' }), {
      toolNameOverride: 'duplicate',
    });

    await expectRejectedBeforeModelRequest({
      handoffs: [first, second],
      expectedMessage: `Duplicate enabled tool name found: 'duplicate' (2 handoffs). ${remediation}`,
    });
  });

  it('rejects mixed enabled function tool and handoff names before a model request', async () => {
    const duplicateHandoff = handoff(new Agent({ name: 'Target' }), {
      toolNameOverride: 'duplicate',
    });

    await expectRejectedBeforeModelRequest({
      tools: [functionTool('duplicate')],
      handoffs: [duplicateHandoff],
      expectedMessage: `Duplicate enabled tool name found: 'duplicate' (function tool and handoff). ${remediation}`,
    });
  });

  it('warns by default and exposes only the last function tool', async () => {
    const model = new RecordingModel([TEST_MODEL_RESPONSE_BASIC]);
    const agent = new Agent({
      name: 'Last function wins agent',
      model,
      tools: [
        functionTool('duplicate', true, 'First tool'),
        functionTool('duplicate', true, 'Second tool'),
      ],
    });

    await run(agent, 'hello');

    expect(model.requests).toHaveLength(1);
    expect(model.requests[0]?.tools).toEqual([
      expect.objectContaining({
        name: 'duplicate',
        description: 'Second tool',
      }),
    ]);
    expect(logger.warn).toHaveBeenCalledWith(
      `Duplicate enabled tool name found: 'duplicate' (2 function tools). ${remediation} Only the current dispatch winner will be exposed.`,
    );
  });

  it('dispatches the same last function tool that was exposed to the model', async () => {
    const firstExecute = vi.fn(async () => 'first');
    const secondExecute = vi.fn(async () => 'second');
    const model = new RecordingModel([
      {
        output: [
          {
            ...TEST_MODEL_FUNCTION_CALL,
            name: 'duplicate',
            arguments: '{}',
          },
        ],
        usage: new Usage(),
      },
      TEST_MODEL_RESPONSE_BASIC,
    ]);
    const agent = new Agent({
      name: 'Dispatch winner agent',
      model,
      tools: [
        functionTool('duplicate', true, 'First tool', firstExecute),
        functionTool('duplicate', true, 'Second tool', secondExecute),
      ],
    });

    await run(agent, 'hello');

    expect(model.requests[0]?.tools).toEqual([
      expect.objectContaining({ description: 'Second tool' }),
    ]);
    expect(firstExecute).not.toHaveBeenCalled();
    expect(secondExecute).toHaveBeenCalledTimes(1);
  });

  it('warns by default and gives handoffs priority over function tools', async () => {
    const firstHandoff = handoff(new Agent({ name: 'First target' }), {
      toolNameOverride: 'duplicate',
      toolDescriptionOverride: 'First handoff',
    });
    const secondHandoff = handoff(new Agent({ name: 'Second target' }), {
      toolNameOverride: 'duplicate',
      toolDescriptionOverride: 'Second handoff',
    });
    const model = new RecordingModel([TEST_MODEL_RESPONSE_BASIC]);
    const agent = new Agent({
      name: 'Handoff wins agent',
      model,
      tools: [functionTool('duplicate')],
      handoffs: [firstHandoff, secondHandoff],
    });

    await run(agent, 'hello');

    expect(model.requests).toHaveLength(1);
    expect(model.requests[0]?.tools).toEqual([]);
    expect(model.requests[0]?.handoffs).toEqual([
      expect.objectContaining({
        toolName: 'duplicate',
        toolDescription: 'Second handoff',
      }),
    ]);
    expect(logger.warn).toHaveBeenCalledWith(
      `Duplicate enabled tool name found: 'duplicate' (function tool and 2 handoffs). ${remediation} Only the current dispatch winner will be exposed.`,
    );
  });

  it('ignores disabled function tools and handoffs when checking names', async () => {
    const enabledHandoff = handoff(new Agent({ name: 'Enabled target' }), {
      toolNameOverride: 'handoff_duplicate',
    });
    const disabledHandoff = handoff(new Agent({ name: 'Disabled target' }), {
      toolNameOverride: 'handoff_duplicate',
      isEnabled: false,
    });
    const model = new RecordingModel([TEST_MODEL_RESPONSE_BASIC]);
    const agent = new Agent({
      name: 'Filtered capabilities agent',
      model,
      tools: [
        functionTool('function_duplicate'),
        functionTool('function_duplicate', false),
      ],
      handoffs: [enabledHandoff, disabledHandoff],
    });

    await run(agent, 'hello');

    expect(model.requests).toHaveLength(1);
    expect(logger.warn).not.toHaveBeenCalled();
    expect(model.requests[0]?.tools).toEqual([
      expect.objectContaining({ name: 'function_duplicate' }),
    ]);
    expect(model.requests[0]?.handoffs).toEqual([
      expect.objectContaining({ toolName: 'handoff_duplicate' }),
    ]);
  });

  it('keeps equal bare names distinct when namespaces disambiguate routing', async () => {
    const [crmLookup] = toolNamespace({
      name: 'crm',
      description: 'CRM tools',
      tools: [functionTool('lookup')],
    });
    const [billingLookup] = toolNamespace({
      name: 'billing',
      description: 'Billing tools',
      tools: [functionTool('lookup')],
    });
    const model = new RecordingModel([TEST_MODEL_RESPONSE_BASIC]);
    const dottedHandoff = handoff(new Agent({ name: 'Dotted target' }), {
      toolNameOverride: 'crm.lookup',
    });
    const agent = new Agent({
      name: 'Namespaced capabilities agent',
      model,
      tools: [functionTool('lookup'), crmLookup!, billingLookup!],
      handoffs: [dottedHandoff],
    });

    await run(agent, 'hello');

    expect(model.requests).toHaveLength(1);
    expect(model.requests[0]?.tools).toEqual([
      expect.objectContaining({ name: 'lookup' }),
      expect.objectContaining({ name: 'lookup', namespace: 'crm' }),
      expect.objectContaining({ name: 'lookup', namespace: 'billing' }),
    ]);
    expect(model.requests[0]?.handoffs).toEqual([
      expect.objectContaining({ toolName: 'crm.lookup' }),
    ]);
  });

  it('uses the same pre-request validation for streaming runs', async () => {
    const duplicateHandoff = handoff(new Agent({ name: 'Stream target' }), {
      toolNameOverride: 'duplicate',
    });
    const model = new FakeModel([TEST_MODEL_RESPONSE_BASIC]);
    const getResponse = vi.spyOn(model, 'getResponse');
    const getStreamedResponse = vi.spyOn(model, 'getStreamedResponse');
    const agent = new Agent({
      name: 'Streaming collision agent',
      model,
      tools: [functionTool('duplicate')],
      handoffs: [duplicateHandoff],
    });

    const result = await run(agent, 'hello', {
      stream: true,
      toolNameCollisionPolicy: 'error',
    });

    await expect(result.completed).rejects.toMatchObject({
      name: UserError.name,
      message: `Duplicate enabled tool name found: 'duplicate' (function tool and handoff). ${remediation}`,
    });
    expect(getResponse).not.toHaveBeenCalled();
    expect(getStreamedResponse).not.toHaveBeenCalled();
  });

  it('redacts colliding names while keeping remediation actionable', async () => {
    const redaction = vi
      .spyOn(logger, 'dontLogToolData', 'get')
      .mockReturnValue(true);
    try {
      await expectRejectedBeforeModelRequest({
        tools: [
          functionTool('sensitive_duplicate'),
          functionTool('sensitive_duplicate'),
        ],
        expectedMessage:
          'Duplicate enabled function tool or handoff names found. ' +
          remediation,
        policy: 'error',
      });
    } finally {
      redaction.mockRestore();
    }
  });

  it('redacts warning details while keeping remediation actionable', async () => {
    vi.spyOn(logger, 'dontLogToolData', 'get').mockReturnValue(true);
    const model = new RecordingModel([TEST_MODEL_RESPONSE_BASIC]);
    const agent = new Agent({
      name: 'Redacted warning agent',
      model,
      tools: [
        functionTool('sensitive_duplicate'),
        functionTool('sensitive_duplicate'),
      ],
    });

    await run(agent, 'hello');

    expect(model.requests[0]?.tools).toHaveLength(1);
    expect(logger.warn).toHaveBeenCalledWith(
      'Tool name collision detected. Assign unique routed tool names or enable tool data logging for details. Only the current dispatch winner will be exposed.',
    );
  });
});
