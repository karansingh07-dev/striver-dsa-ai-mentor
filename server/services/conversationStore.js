import { randomUUID } from "crypto";

const conversations = new Map();

export function createConversation(mode = "explain") {
  const id = randomUUID();
  conversations.set(id, {
    id,
    mode,
    messages: [],
    state: {},
    createdAt: Date.now(),
    updatedAt: Date.now(),
  });
  return getConversation(id);
}

export function getConversation(id) {
  return conversations.get(id) || null;
}

export function updateConversation(id, updater) {
  const convo = conversations.get(id);
  if (!convo) return null;
  const updated = typeof updater === "function" ? updater(convo) : updater;
  conversations.set(id, { ...convo, ...updated, updatedAt: Date.now() });
  return getConversation(id);
}

export function deleteConversation(id) {
  conversations.delete(id);
}

export function getOrCreateConversation(id, mode = "explain") {
  if (id && conversations.has(id)) {
    return getConversation(id);
  }
  return createConversation(mode);
}

export function buildHistory(conversation, limit = 6) {
  if (!conversation) return [];
  return conversation.messages.slice(-limit);
}
