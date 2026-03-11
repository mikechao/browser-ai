import { describe, it, expect, beforeEach, vi } from "vitest";
import { SessionManager } from "../src/chat/session-manager";
import type { DownloadProgressCallback } from "@browser-ai/shared";

describe("SessionManager", () => {
  beforeEach(() => {
    // Reset the global mock before each test
    vi.stubGlobal("LanguageModel", {
      availability: vi.fn().mockResolvedValue("available"),
      create: vi.fn().mockResolvedValue({
        prompt: vi.fn(),
        destroy: vi.fn(),
      }),
    });
  });

  describe("constructor", () => {
    it("should create a session manager with base options", () => {
      const manager = new SessionManager({
        temperature: 0.7,
        topK: 40,
      });

      expect(manager).toBeInstanceOf(SessionManager);
    });

    it("should filter out custom provider options", () => {
      const manager = new SessionManager({
        temperature: 0.7,
        parallelToolExecution: true,
        debugToolCalls: true,
      } as any);

      expect(manager).toBeInstanceOf(SessionManager);
      expect(manager.getCurrentSession()).toBeNull();
    });
  });

  describe("checkAvailability", () => {
    it("should return unavailable when LanguageModel is not defined", async () => {
      vi.stubGlobal("LanguageModel", undefined);
      const manager = new SessionManager({});
      const availability = await manager.checkAvailability();

      expect(availability).toBe("unavailable");
    });

    it("should return availability from LanguageModel API", async () => {
      const mockAvailability = vi.fn().mockResolvedValue("available");
      vi.stubGlobal("LanguageModel", {
        availability: mockAvailability,
        create: vi.fn(),
      });

      const manager = new SessionManager({});
      const availability = await manager.checkAvailability();

      expect(availability).toBe("available");
      expect(mockAvailability).toHaveBeenCalledTimes(1);
    });

    it("should return downloadable when model needs download", async () => {
      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("downloadable"),
        create: vi.fn(),
      });

      const manager = new SessionManager({});
      const availability = await manager.checkAvailability();

      expect(availability).toBe("downloadable");
    });
  });

  describe("getSession", () => {
    it("should throw error when LanguageModel is undefined", async () => {
      vi.stubGlobal("LanguageModel", undefined);
      const manager = new SessionManager({});

      await expect(manager.getSession()).rejects.toThrow(
        "Prompt API is not available",
      );
    });

    it("should throw error when model is unavailable", async () => {
      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("unavailable"),
        create: vi.fn(),
      });

      const manager = new SessionManager({});

      await expect(manager.getSession()).rejects.toThrow(
        "Built-in model not available",
      );
    });

    it("should create a session when available", async () => {
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      };
      const mockCreate = vi.fn().mockResolvedValue(mockSession);

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({ temperature: 0.7 });
      const session = await manager.getSession();

      expect(session).toBe(mockSession);
      expect(mockCreate).toHaveBeenCalledWith({ temperature: 0.7 });
    });

    it("should reuse existing session", async () => {
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      };
      const mockCreate = vi.fn().mockResolvedValue(mockSession);

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({});
      const session1 = await manager.getSession();
      const session2 = await manager.getSession();

      expect(session1).toBe(session2);
      expect(mockCreate).toHaveBeenCalledTimes(1);
    });

    it("should handle systemMessage option", async () => {
      const mockCreate = vi.fn().mockResolvedValue({
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      });

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({});
      await manager.getSession({
        systemMessage: "You are a helpful assistant",
      });

      expect(mockCreate).toHaveBeenCalledWith({
        initialPrompts: [
          { role: "system", content: "You are a helpful assistant" },
        ],
      });
    });

    it("should handle expectedInputs option", async () => {
      const mockCreate = vi.fn().mockResolvedValue({
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      });

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({});
      await manager.getSession({
        expectedInputs: [{ type: "image" }, { type: "text" }],
      });

      expect(mockCreate).toHaveBeenCalledWith({
        expectedInputs: [{ type: "image" }, { type: "text" }],
      });
    });

    it("should preserve base expectedInputs when runtime expectedInputs are not provided", async () => {
      const mockCreate = vi.fn().mockResolvedValue({
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      });

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({
        expectedInputs: [{ type: "image" }],
      });
      await manager.getSession({});

      expect(mockCreate).toHaveBeenCalledWith({
        expectedInputs: [{ type: "image" }],
      });
    });

    it("should merge runtime expectedInputs with base expectedInputs", async () => {
      const mockCreate = vi.fn().mockResolvedValue({
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      });

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({
        expectedInputs: [{ type: "image" }],
      });
      await manager.getSession({
        expectedInputs: [{ type: "audio" }],
      });

      expect(mockCreate).toHaveBeenCalledWith({
        expectedInputs: [{ type: "image" }, { type: "audio" }],
      });
    });

    it("should merge metadata for duplicate expectedInputs types", async () => {
      const mockCreate = vi.fn().mockResolvedValue({
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      });

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({
        expectedInputs: [{ type: "text", languages: ["en", "da"] }],
      });
      await manager.getSession({
        expectedInputs: [{ type: "text", languages: ["en", "fr"] }],
      });

      expect(mockCreate).toHaveBeenCalledWith({
        expectedInputs: [{ type: "text", languages: ["en", "da", "fr"] }],
      });
    });

    it("should treat empty runtime expectedInputs as no-op", async () => {
      const mockCreate = vi.fn().mockResolvedValue({
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      });

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({
        expectedInputs: [{ type: "audio" }],
      });
      await manager.getSession({
        expectedInputs: [],
      });

      expect(mockCreate).toHaveBeenCalledWith({
        expectedInputs: [{ type: "audio" }],
      });
    });

    it("should merge base, runtime, and initialPrompts inferred expectedInputs", async () => {
      const mockCreate = vi.fn().mockResolvedValue({
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      });

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({
        expectedInputs: [{ type: "text" }],
        initialPrompts: [
          {
            role: "user",
            content: [
              { type: "text", value: "Describe this audio" },
              { type: "audio", value: new Uint8Array([1, 2, 3]) },
            ],
          },
        ],
      });
      await manager.getSession({
        expectedInputs: [{ type: "image" }],
      });

      expect(mockCreate).toHaveBeenCalledWith({
        expectedInputs: [
          { type: "text" },
          { type: "image" },
          { type: "audio" },
        ],
        initialPrompts: [
          {
            role: "user",
            content: [
              { type: "text", value: "Describe this audio" },
              { type: "audio", value: new Uint8Array([1, 2, 3]) },
            ],
          },
        ],
      });
    });

    it("should handle download progress callback", async () => {
      const progressCallback: DownloadProgressCallback = vi.fn();
      const mockCreate = vi.fn().mockResolvedValue({
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      });

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({});
      await manager.getSession({
        onDownloadProgress: progressCallback,
      });

      expect(mockCreate).toHaveBeenCalledTimes(1);
      const createArgs = mockCreate.mock.calls[0][0];
      expect(createArgs.monitor).toBeInstanceOf(Function);
    });

    it("should preserve structured system initialPrompts content when merging runtime systemMessage", async () => {
      const mockCreate = vi.fn().mockResolvedValue({
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      });

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({
        initialPrompts: [
          {
            role: "system",
            content: [{ type: "text", value: "Talk like a pirate" }],
          },
        ],
      });
      await manager.getSession({
        systemMessage: "Tool calling instructions",
      });

      expect(mockCreate).toHaveBeenCalledWith({
        initialPrompts: [
          {
            role: "system",
            content: [
              { type: "text", value: "Talk like a pirate" },
              { type: "text", value: "\n\nTool calling instructions" },
            ],
          },
        ],
      });
    });

    it("should merge constructor initialPrompts system entry with runtime systemMessage", async () => {
      const mockCreate = vi.fn().mockResolvedValue({
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      });

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({
        initialPrompts: [{ role: "system", content: "Talk like a pirate" }],
      });
      await manager.getSession({
        systemMessage: "Tool calling instructions",
      });

      expect(mockCreate).toHaveBeenCalledWith({
        initialPrompts: [
          {
            role: "system",
            content: "Talk like a pirate\n\nTool calling instructions",
          },
        ],
      });
    });

    it("should preserve non-system initialPrompts entries alongside merged system entry", async () => {
      const mockCreate = vi.fn().mockResolvedValue({
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      });

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({
        initialPrompts: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: "What is the capital of Italy?" },
          { role: "assistant", content: "The capital of Italy is Rome." },
        ],
      });
      await manager.getSession({
        systemMessage: "Tool calling instructions",
      });

      expect(mockCreate).toHaveBeenCalledWith({
        initialPrompts: [
          {
            role: "system",
            content:
              "You are a helpful assistant.\n\nTool calling instructions",
          },
          { role: "user", content: "What is the capital of Italy?" },
          { role: "assistant", content: "The capital of Italy is Rome." },
        ],
      });
    });

    it("should preserve non-system initialPrompts entries when there is no existing system entry", async () => {
      const mockCreate = vi.fn().mockResolvedValue({
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      });

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({
        initialPrompts: [
          { role: "user", content: "What is the capital of Italy?" },
          { role: "assistant", content: "The capital of Italy is Rome." },
        ],
      });
      await manager.getSession({
        systemMessage: "Tool calling instructions",
      });

      expect(mockCreate).toHaveBeenCalledWith({
        initialPrompts: [
          { role: "system", content: "Tool calling instructions" },
          { role: "user", content: "What is the capital of Italy?" },
          { role: "assistant", content: "The capital of Italy is Rome." },
        ],
      });
    });

    it("should merge base options with request options", async () => {
      const mockCreate = vi.fn().mockResolvedValue({
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      });

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({
        temperature: 0.7,
        topK: 40,
      });
      await manager.getSession({
        temperature: 0.9,
      });

      expect(mockCreate).toHaveBeenCalledWith({
        temperature: 0.9,
        topK: 40,
      });
    });
  });

  describe("createSessionWithProgress", () => {
    it("should create session with progress callback", async () => {
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      };
      const mockCreate = vi.fn().mockResolvedValue(mockSession);
      const progressCallback: DownloadProgressCallback = vi.fn();

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({});
      const session = await manager.createSessionWithProgress(progressCallback);

      expect(session).toBe(mockSession);
      expect(mockCreate).toHaveBeenCalledTimes(1);
    });

    it("should work without progress callback", async () => {
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      };
      const mockCreate = vi.fn().mockResolvedValue(mockSession);

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({});
      const session = await manager.createSessionWithProgress();

      expect(session).toBe(mockSession);
    });
  });

  describe("getCurrentSession", () => {
    it("should return null when no session exists", () => {
      const manager = new SessionManager({});
      expect(manager.getCurrentSession()).toBeNull();
    });

    it("should return the current session after creation", async () => {
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      };

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: vi.fn().mockResolvedValue(mockSession),
      });

      const manager = new SessionManager({});
      await manager.getSession();

      expect(manager.getCurrentSession()).toBe(mockSession);
    });
  });

  describe("destroySession", () => {
    it("should destroy the current session", async () => {
      const mockDestroy = vi.fn();
      const mockSession = {
        prompt: vi.fn(),
        destroy: mockDestroy,
        addEventListener: vi.fn(),
      };

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: vi.fn().mockResolvedValue(mockSession),
      });

      const manager = new SessionManager({});
      await manager.getSession();

      manager.destroySession();

      expect(mockDestroy).toHaveBeenCalledTimes(1);
      expect(manager.getCurrentSession()).toBeNull();
    });

    it("should handle session without destroy method", async () => {
      const mockSession = {
        prompt: vi.fn(),
        addEventListener: vi.fn(),
      };

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: vi.fn().mockResolvedValue(mockSession),
      });

      const manager = new SessionManager({});
      await manager.getSession();

      expect(() => manager.destroySession()).not.toThrow();
      expect(manager.getCurrentSession()).toBeNull();
    });

    it("should allow creating a new session after destruction", async () => {
      const mockSession1 = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      };
      const mockSession2 = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
      };
      const mockCreate = vi
        .fn()
        .mockResolvedValueOnce(mockSession1)
        .mockResolvedValueOnce(mockSession2);

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: mockCreate,
      });

      const manager = new SessionManager({});
      const session1 = await manager.getSession();
      manager.destroySession();
      const session2 = await manager.getSession();

      expect(session1).not.toBe(session2);
      expect(mockCreate).toHaveBeenCalledTimes(2);
    });
  });

  describe("getContextWindow", () => {
    it("should return undefined when no session exists", () => {
      const manager = new SessionManager({});
      expect(manager.getContextWindow()).toBeUndefined();
    });

    it("should return the context window from the session", async () => {
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
        contextWindow: 4096,
      };

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: vi.fn().mockResolvedValue(mockSession),
      });

      const manager = new SessionManager({});
      await manager.getSession();

      expect(manager.getContextWindow()).toBe(4096);
    });

    it("should fall back to inputQuota on older Chrome builds", async () => {
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
        inputQuota: 2048,
      };

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: vi.fn().mockResolvedValue(mockSession),
      });

      const manager = new SessionManager({});
      await manager.getSession();

      expect(manager.getContextWindow()).toBe(2048);
    });
  });

  describe("getContextUsage", () => {
    it("should return undefined when no session exists", () => {
      const manager = new SessionManager({});
      expect(manager.getContextUsage()).toBeUndefined();
    });

    it("should return the context usage from the session", async () => {
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
        contextUsage: 128,
      };

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: vi.fn().mockResolvedValue(mockSession),
      });

      const manager = new SessionManager({});
      await manager.getSession();

      expect(manager.getContextUsage()).toBe(128);
    });

    it("should fall back to inputUsage on older Chrome builds", async () => {
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: vi.fn(),
        inputUsage: 64,
      };

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: vi.fn().mockResolvedValue(mockSession),
      });

      const manager = new SessionManager({});
      await manager.getSession();

      expect(manager.getContextUsage()).toBe(64);
    });
  });

  describe("onContextOverflow", () => {
    it("should attach custom onContextOverflow handler from getSession options", async () => {
      const mockAddEventListener = vi.fn();
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: mockAddEventListener,
        oncontextoverflow: null,
      };

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: vi.fn().mockResolvedValue(mockSession),
      });

      const onContextOverflow = vi.fn();
      const manager = new SessionManager({});
      await manager.getSession({ onContextOverflow });

      expect(mockAddEventListener).toHaveBeenCalledWith(
        "contextoverflow",
        onContextOverflow,
      );
    });

    it("should attach custom onContextOverflow handler from base options", async () => {
      const mockAddEventListener = vi.fn();
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: mockAddEventListener,
        oncontextoverflow: null,
      };

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: vi.fn().mockResolvedValue(mockSession),
      });

      const onContextOverflow = vi.fn();
      const manager = new SessionManager({ onContextOverflow } as any);
      await manager.getSession();

      expect(mockAddEventListener).toHaveBeenCalledWith(
        "contextoverflow",
        onContextOverflow,
      );
    });

    it("should prefer request-level onContextOverflow over base options", async () => {
      const mockAddEventListener = vi.fn();
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: mockAddEventListener,
        oncontextoverflow: null,
      };

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: vi.fn().mockResolvedValue(mockSession),
      });

      const baseOnContextOverflow = vi.fn();
      const requestOnContextOverflow = vi.fn();
      const manager = new SessionManager({
        onContextOverflow: baseOnContextOverflow,
      } as any);
      await manager.getSession({ onContextOverflow: requestOnContextOverflow });

      expect(mockAddEventListener).toHaveBeenCalledWith(
        "contextoverflow",
        requestOnContextOverflow,
      );
      expect(mockAddEventListener).not.toHaveBeenCalledWith(
        "contextoverflow",
        baseOnContextOverflow,
      );
    });

    it("should accept deprecated onQuotaOverflow as a fallback alias", async () => {
      const mockAddEventListener = vi.fn();
      // No oncontextoverflow property → exercises the quotaoverflow fallback path
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: mockAddEventListener,
      };

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: vi.fn().mockResolvedValue(mockSession),
      });

      const onQuotaOverflow = vi.fn();
      const manager = new SessionManager({});
      await manager.getSession({ onQuotaOverflow });

      expect(mockAddEventListener).toHaveBeenCalledWith(
        "quotaoverflow",
        onQuotaOverflow,
      );
    });

    it("should attach default console.warn handler when no handler provided", async () => {
      const mockAddEventListener = vi.fn();
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: mockAddEventListener,
        oncontextoverflow: null,
      };

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: vi.fn().mockResolvedValue(mockSession),
      });

      const manager = new SessionManager({});
      await manager.getSession();

      expect(mockAddEventListener).toHaveBeenCalledWith(
        "contextoverflow",
        expect.any(Function),
      );
    });

    it("should invoke default handler and log warning when context overflow event fires", async () => {
      let capturedHandler: ((event: Event) => void) | null = null;
      const mockAddEventListener = vi.fn((event, handler) => {
        if (event === "contextoverflow") {
          capturedHandler = handler;
        }
      });
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: mockAddEventListener,
        oncontextoverflow: null,
      };

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: vi.fn().mockResolvedValue(mockSession),
      });

      const consoleWarnSpy = vi
        .spyOn(console, "warn")
        .mockImplementation(() => {});

      const manager = new SessionManager({});
      await manager.getSession();

      expect(capturedHandler).not.toBeNull();
      capturedHandler!(new Event("contextoverflow"));

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        "Model context window exceeded. Consider handling the 'contextoverflow' event.",
      );

      consoleWarnSpy.mockRestore();
    });

    it("should invoke custom onContextOverflow handler when context overflow event fires", async () => {
      let capturedHandler: ((event: Event) => void) | null = null;
      const mockAddEventListener = vi.fn((event, handler) => {
        if (event === "contextoverflow") {
          capturedHandler = handler;
        }
      });
      const mockSession = {
        prompt: vi.fn(),
        destroy: vi.fn(),
        addEventListener: mockAddEventListener,
        oncontextoverflow: null,
      };

      vi.stubGlobal("LanguageModel", {
        availability: vi.fn().mockResolvedValue("available"),
        create: vi.fn().mockResolvedValue(mockSession),
      });

      const onContextOverflow = vi.fn();
      const manager = new SessionManager({});
      await manager.getSession({ onContextOverflow });

      expect(capturedHandler).not.toBeNull();
      const mockEvent = new Event("contextoverflow");
      capturedHandler!(mockEvent);

      expect(onContextOverflow).toHaveBeenCalledTimes(1);
      expect(onContextOverflow).toHaveBeenCalledWith(mockEvent);
    });
  });
});
