import type { SampleConversation } from "./types";

export const samples: SampleConversation[] = [
  {
    name: "Cyberstalking (proposal example)",
    messages: [
      { speaker: "User B", text: "Please stop messaging me." },
      { speaker: "User A", text: "No." },
      { speaker: "User B", text: "I already blocked you." },
      { speaker: "User A", text: "I'll make another account." },
    ],
  },
  {
    name: "Taglish: stalking and privacy",
    messages: [
      { speaker: "User A", text: "I will keep messaging you kahit hindi ka sumasagot. 😠" },
      { speaker: "User B", text: "Sinabi ko nang tumigil ka sa pagkontak sa akin. Please stop. 😟" },
      { speaker: "User A", text: "I will message you from another account kapag blocked ako. 😈" },
      { speaker: "User B", text: "Why are you still following my accounts? Huwag mo na akong i-follow." },
      { speaker: "User A", text: "I monitor your posts para malaman kung nasaan ka. 👀" },
      { speaker: "User B", text: "Please stop monitoring me. Nakakatakot na ito. 😟" },
    ],
  },
  {
    name: "Unwanted sexual remarks",
    messages: [
      { speaker: "User A", text: "Hello, I saw your profile and I wanted to say hi." },
      { speaker: "User A", text: "Can you send me a picture of yourself sexy 🥵" },
      { speaker: "User B", text: "No, please stop messaging me." },
      { speaker: "User A", text: "Lol. I’ve asked several times already. Malandi ka!" },
    ],
  },
  {
    name: "Non-harassment",
    messages: [
      { speaker: "User A", text: "Hi, are you free later for the group project?" },
      { speaker: "User B", text: "Yes, after class is fine." },
      { speaker: "User A", text: "Thanks, I will send the slides." },
    ],
  },
];
