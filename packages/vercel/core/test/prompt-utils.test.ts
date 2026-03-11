import { describe, it, expect } from "vitest";
import {
  getMultimodalInfo,
  getMultimodalTypesFromInitialPrompts,
  prependSystemPromptToMessages,
} from "../src/utils/prompt-utils";
import type { LanguageModelV3Prompt } from "@ai-sdk/provider";

describe("prompt-utils", () => {
  describe("getMultimodalInfo", () => {
    it("returns hasMultiModalInput=false and no expectedInputs for text-only prompt", () => {
      const prompt: LanguageModelV3Prompt = [
        { role: "user", content: [{ type: "text", text: "Hello" }] },
      ];
      expect(getMultimodalInfo(prompt)).toEqual({
        hasMultiModalInput: false,
        expectedInputs: undefined,
      });
    });

    it("detects image file", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            { type: "text", text: "What's in this image?" },
            { type: "file", data: new Uint8Array(), mediaType: "image/png" },
          ],
        },
      ];
      expect(getMultimodalInfo(prompt)).toEqual({
        hasMultiModalInput: true,
        expectedInputs: [{ type: "image" }],
      });
    });

    it("detects audio file", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            { type: "file", data: new Uint8Array(), mediaType: "audio/wav" },
          ],
        },
      ];
      expect(getMultimodalInfo(prompt)).toEqual({
        hasMultiModalInput: true,
        expectedInputs: [{ type: "audio" }],
      });
    });

    it("detects both image and audio when both present", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            { type: "file", data: new Uint8Array(), mediaType: "image/png" },
            { type: "file", data: new Uint8Array(), mediaType: "audio/wav" },
          ],
        },
      ];
      const { hasMultiModalInput, expectedInputs } = getMultimodalInfo(prompt);
      expect(hasMultiModalInput).toBe(true);
      expect(expectedInputs).toHaveLength(2);
      expect(expectedInputs).toContainEqual({ type: "image" });
      expect(expectedInputs).toContainEqual({ type: "audio" });
    });

    it("deduplicates multiple images", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            { type: "file", data: new Uint8Array(), mediaType: "image/png" },
            { type: "file", data: new Uint8Array(), mediaType: "image/jpeg" },
          ],
        },
      ];
      expect(getMultimodalInfo(prompt)).toEqual({
        hasMultiModalInput: true,
        expectedInputs: [{ type: "image" }],
      });
    });

    it("ignores files with unknown media types", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            {
              type: "file",
              data: new Uint8Array(),
              mediaType: "application/pdf",
            },
          ],
        },
      ];
      expect(getMultimodalInfo(prompt)).toEqual({
        hasMultiModalInput: false,
        expectedInputs: undefined,
      });
    });

    it("ignores files on assistant messages (only checks user messages)", () => {
      const prompt: LanguageModelV3Prompt = [
        { role: "user", content: [{ type: "text", text: "Hello" }] },
        {
          role: "assistant",
          content: [
            {
              type: "file" as any,
              data: new Uint8Array(),
              mediaType: "image/png",
            },
          ],
        },
      ];
      expect(getMultimodalInfo(prompt)).toEqual({
        hasMultiModalInput: false,
        expectedInputs: undefined,
      });
    });
  });

  describe("getMultimodalTypesFromInitialPrompts", () => {
    it("returns empty set for empty array", () => {
      expect(getMultimodalTypesFromInitialPrompts([])).toEqual(new Set());
    });

    it("returns empty set for string content (system prompt)", () => {
      const prompts = [
        { role: "system", content: "You are helpful." },
      ] as LanguageModelMessage[];
      expect(getMultimodalTypesFromInitialPrompts(prompts)).toEqual(new Set());
    });

    it("detects image type", () => {
      const prompts = [
        {
          role: "user",
          content: [
            { type: "text", value: "Describe this." },
            { type: "image", value: new Uint8Array([1, 2, 3]) },
          ],
        },
        { role: "assistant", content: "It is a lighthouse." },
      ] as LanguageModelMessage[];
      expect(getMultimodalTypesFromInitialPrompts(prompts)).toEqual(
        new Set(["image"]),
      );
    });

    it("detects audio type", () => {
      const prompts = [
        {
          role: "user",
          content: [{ type: "audio", value: new Uint8Array([4, 5, 6]) }],
        },
      ] as LanguageModelMessage[];
      expect(getMultimodalTypesFromInitialPrompts(prompts)).toEqual(
        new Set(["audio"]),
      );
    });

    it("detects and deduplicates both image and audio", () => {
      const prompts = [
        {
          role: "user",
          content: [
            { type: "image", value: new Uint8Array([1]) },
            { type: "image", value: new Uint8Array([2]) },
            { type: "audio", value: new Uint8Array([3]) },
          ],
        },
      ] as LanguageModelMessage[];
      expect(getMultimodalTypesFromInitialPrompts(prompts)).toEqual(
        new Set(["image", "audio"]),
      );
    });

    it("ignores text-only content arrays", () => {
      const prompts = [
        {
          role: "user",
          content: [{ type: "text", value: "Hello" }],
        },
      ] as LanguageModelMessage[];
      expect(getMultimodalTypesFromInitialPrompts(prompts)).toEqual(new Set());
    });
  });

  describe("prependSystemPromptToMessages", () => {
    it("returns messages unchanged for empty system prompt", () => {
      const messages = [
        { role: "user", content: "Hello" },
      ] as LanguageModelMessage[];

      expect(prependSystemPromptToMessages(messages, "")).toEqual(messages);
      expect(prependSystemPromptToMessages(messages, "   ")).toEqual(messages);
    });

    it("prepends system prompt to string content user message", () => {
      const messages = [
        { role: "user", content: "Hello" },
      ] as LanguageModelMessage[];

      const result = prependSystemPromptToMessages(messages, "You are helpful");

      expect(result).toHaveLength(1);
      expect(result[0]).toMatchObject({
        role: "user",
        content: "You are helpful\n\nHello",
      });
    });

    it("prepends system prompt to array content user message", () => {
      const messages = [
        {
          role: "user",
          content: [{ type: "text", value: "Hello" }],
        },
      ] as LanguageModelMessage[];

      const result = prependSystemPromptToMessages(messages, "You are helpful");

      expect(result).toHaveLength(1);
      expect(result[0].content).toEqual([
        { type: "text", value: "You are helpful\n\n" },
        { type: "text", value: "Hello" },
      ]);
    });

    it("creates new user message if no user messages exist", () => {
      const messages = [
        { role: "assistant", content: "Hi there" },
      ] as LanguageModelMessage[];

      const result = prependSystemPromptToMessages(messages, "You are helpful");

      expect(result).toHaveLength(2);
      expect(result[0]).toMatchObject({
        role: "user",
        content: "You are helpful",
      });
      expect(result[1]).toMatchObject({
        role: "assistant",
        content: "Hi there",
      });
    });

    it("prepends to first user message when multiple exist", () => {
      const messages = [
        { role: "assistant", content: "Hello" },
        { role: "user", content: "First user" },
        { role: "user", content: "Second user" },
      ] as LanguageModelMessage[];

      const result = prependSystemPromptToMessages(messages, "Be helpful");

      expect(result).toHaveLength(3);
      expect(result[1]).toMatchObject({
        role: "user",
        content: "Be helpful\n\nFirst user",
      });
      expect(result[2]).toMatchObject({
        role: "user",
        content: "Second user",
      });
    });

    it("does not mutate original messages array", () => {
      const messages = [
        { role: "user", content: "Hello" },
      ] as LanguageModelMessage[];

      const original = JSON.stringify(messages);
      prependSystemPromptToMessages(messages, "System");

      expect(JSON.stringify(messages)).toBe(original);
    });
  });
});
