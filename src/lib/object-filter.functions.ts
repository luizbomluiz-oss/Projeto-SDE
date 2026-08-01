import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { GoogleGenAI } from "@google/genai";

const Input = z.object({
  /** data URL of the reference object image */
  reference: z.string().min(20),
  /** data URL of the captured frame */
  frame: z.string().min(20),
  /** optional textual hint about the object */
  hint: z.string().max(200).optional(),
});

export type ObjectFilterResult = { hasObject: boolean; confidence: number; note: string };

function parseDataUrl(dataUrl: string) {
  const match = dataUrl.match(/^data:(image\/[a-zA-Z0-9+.-]+);base64,(.+)$/);
  if (!match) return null;
  return { mimeType: match[1], data: match[2] };
}

/**
 * Uses Gemini AI vision (or Lovable gateway fallback) to decide whether the person in the captured frame
 * carries/wears the reference object.
 */
export const detectObjectInFrame = createServerFn({ method: "POST" })
  .inputValidator((input: unknown) => Input.parse(input))
  .handler(async ({ data }): Promise<ObjectFilterResult> => {
    const geminiKey = process.env.GEMINI_API_KEY;
    const lovableKey = process.env.LOVABLE_API_KEY;

    if (geminiKey) {
      const ai = new GoogleGenAI({ apiKey: geminiKey });
      const refParsed = parseDataUrl(data.reference);
      const frameParsed = parseDataUrl(data.frame);

      if (!refParsed || !frameParsed) {
        return { hasObject: false, confidence: 0, note: "formato de imagem inválido" };
      }

      try {
        const response = await ai.models.generateContent({
          model: "gemini-2.5-flash",
          contents: [
            {
              role: "user",
              parts: [
                {
                  text:
                    'You compare images. The first image is a reference object. The second image is a video frame with a person. Answer ONLY with compact JSON: {"hasObject":boolean,"confidence":number,"note":string}. hasObject is true when the person in the frame carries, wears or holds the reference object.' +
                    (data.hint ? `\nReference object hint: ${data.hint}` : ""),
                },
                { inlineData: { mimeType: refParsed.mimeType, data: refParsed.data } },
                { inlineData: { mimeType: frameParsed.mimeType, data: frameParsed.data } },
              ],
            },
          ],
          config: {
            responseMimeType: "application/json",
          },
        });

        const raw = response.text ?? "";
        const match = raw.match(/\{[\s\S]*\}/);
        if (!match) return { hasObject: false, confidence: 0, note: "resposta inválida" };
        const parsed = JSON.parse(match[0]) as Partial<ObjectFilterResult>;
        return {
          hasObject: Boolean(parsed.hasObject),
          confidence: Number(parsed.confidence ?? 0),
          note: String(parsed.note ?? ""),
        };
      } catch (err) {
        console.error("Gemini API error:", err);
        throw new Error("AI_ERROR");
      }
    }

    if (!lovableKey) {
      throw new Error("Missing GEMINI_API_KEY or LOVABLE_API_KEY");
    }

    const res = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Lovable-API-Key": lovableKey },
      body: JSON.stringify({
        model: "google/gemini-3.6-flash",
        messages: [
          {
            role: "system",
            content:
              'You compare images. The first image is a reference object. The second image is a video frame with a person. Answer ONLY with compact JSON: {"hasObject":boolean,"confidence":number,"note":string}. hasObject is true when the person in the frame carries, wears or holds the reference object.',
          },
          {
            role: "user",
            content: [
              { type: "text", text: `Reference object${data.hint ? `: ${data.hint}` : ""}` },
              { type: "image_url", image_url: { url: data.reference } },
              { type: "text", text: "Frame with the person:" },
              { type: "image_url", image_url: { url: data.frame } },
            ],
          },
        ],
      }),
    });

    if (res.status === 429) throw new Error("RATE_LIMIT");
    if (res.status === 402) throw new Error("NO_CREDITS");
    if (!res.ok) throw new Error(`AI error ${res.status}: ${await res.text()}`);

    const json = (await res.json()) as { choices?: { message?: { content?: string } }[] };
    const raw = json.choices?.[0]?.message?.content ?? "";
    const match = raw.match(/\{[\s\S]*\}/);
    if (!match) return { hasObject: false, confidence: 0, note: "resposta inválida" };
    try {
      const parsed = JSON.parse(match[0]) as Partial<ObjectFilterResult>;
      return {
        hasObject: Boolean(parsed.hasObject),
        confidence: Number(parsed.confidence ?? 0),
        note: String(parsed.note ?? ""),
      };
    } catch {
      return { hasObject: false, confidence: 0, note: "resposta inválida" };
    }
  });
