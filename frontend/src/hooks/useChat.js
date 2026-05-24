import { useState } from "react";

import { askQuestion } from "../services/chatService";

export default function useChat() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  async function send(question, documentIds = []) {
    const history = messages.map(({ role, content }) => ({ role, content }));
    const userMessage = { id: crypto.randomUUID(), role: "user", content: question };
    const assistantId = crypto.randomUUID();
    setMessages((current) => [...current, userMessage, { id: assistantId, role: "assistant", content: "", sources: [] }]);
    setLoading(true);

    const payload = { question, document_ids: documentIds, history };
    try {
      const response = await askQuestion(payload);
      setMessages((current) => current.map((msg) => (msg.id === assistantId ? { ...msg, content: response.answer, sources: response.sources } : msg)));
    } finally {
      setLoading(false);
    }
  }

  return { messages, setMessages, loading, send };
}
