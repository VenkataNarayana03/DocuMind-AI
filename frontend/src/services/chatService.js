import { api } from "./api";

export async function askQuestion(payload) {
  const { data } = await api.post("/chat", payload);
  return data;
}
