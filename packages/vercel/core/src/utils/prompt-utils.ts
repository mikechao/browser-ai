/**
 * Utilities for prompt processing and transformation
 */

import type { LanguageModelV3Prompt } from "@ai-sdk/provider";

/**
 * Detect multimodal content and collect expected input types in a single pass.
 *
 * @param prompt - The prompt to analyze
 * @returns hasMultiModalInput flag and the expectedInputs array (undefined when text-only)
 */
export function getMultimodalInfo(prompt: LanguageModelV3Prompt): {
  hasMultiModalInput: boolean;
  expectedInputs: Array<{ type: "text" | "image" | "audio" }> | undefined;
} {
  const inputs = new Set<"image" | "audio">();

  for (const message of prompt) {
    if (message.role === "user") {
      for (const part of message.content) {
        if (part.type === "file") {
          if (part.mediaType?.startsWith("image/")) {
            inputs.add("image");
          } else if (part.mediaType?.startsWith("audio/")) {
            inputs.add("audio");
          }
        }
      }
    }
  }

  const hasMultiModalInput = inputs.size > 0;
  return {
    hasMultiModalInput,
    expectedInputs: hasMultiModalInput
      ? Array.from(inputs, (type) => ({ type }))
      : undefined,
  };
}

/**
 * Collect image/audio types declared in Prompt API `initialPrompts` content.
 *
 * Prompt API content parts use `type: "image"` and `type: "audio"` directly,
 * unlike AI SDK file parts which use `type: "file"` with a `mediaType`.
 * System prompts have string content and are skipped automatically.
 *
 * @param initialPrompts - The Prompt API initialPrompts array to inspect
 * @returns A Set of multimodal type strings found across all prompts
 */
export function getMultimodalTypesFromInitialPrompts(
  initialPrompts: LanguageModelMessage[],
): Set<"image" | "audio"> {
  const types = new Set<"image" | "audio">();
  for (const prompt of initialPrompts) {
    if (Array.isArray(prompt.content)) {
      for (const part of prompt.content) {
        if (part.type === "image") types.add("image");
        else if (part.type === "audio") types.add("audio");
      }
    }
  }
  return types;
}

/**
 * Prepends a system prompt to the first user message in the conversation.
 *
 * This is necessary because the Prompt API doesn't support separate system messages,
 * so we inject the system prompt into the first user message instead.
 * Creates a shallow copy of messages to avoid mutating the original array.
 *
 * @param messages - The messages array to modify (not mutated, a copy is returned)
 * @param systemPrompt - The system prompt to prepend
 * @returns New messages array with system prompt prepended to first user message
 * @example
 * ```typescript
 * const messages = [{ role: "user", content: "Hello" }];
 * const updated = prependSystemPromptToMessages(messages, "You are a helpful assistant.");
 * // Returns: [{ role: "user", content: "You are a helpful assistant.\n\nHello" }]
 * ```
 */
export function prependSystemPromptToMessages(
  messages: LanguageModelMessage[],
  systemPrompt: string,
): LanguageModelMessage[] {
  if (!systemPrompt.trim()) {
    return messages;
  }

  const prompts = messages.map((message) => ({ ...message }));
  const firstUserIndex = prompts.findIndex(
    (message) => message.role === "user",
  );

  if (firstUserIndex !== -1) {
    const firstUserMessage = prompts[firstUserIndex];

    if (Array.isArray(firstUserMessage.content)) {
      const content = firstUserMessage.content.slice();
      content.unshift({
        type: "text",
        value: `${systemPrompt}\n\n`,
      });
      prompts[firstUserIndex] = {
        ...firstUserMessage,
        content,
      } as LanguageModelMessage;
    } else if (typeof firstUserMessage.content === "string") {
      prompts[firstUserIndex] = {
        ...firstUserMessage,
        content: `${systemPrompt}\n\n${firstUserMessage.content}`,
      } as LanguageModelMessage;
    }
  } else {
    prompts.unshift({
      role: "user",
      content: systemPrompt,
    });
  }

  return prompts;
}
