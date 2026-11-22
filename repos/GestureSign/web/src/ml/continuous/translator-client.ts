/**
 * translator-client.ts
 * Client for Glossa-BART translation server.
 * Handles server online/offline status gracefully and provides optional Text-to-Speech.
 */

export interface TranslationResponse {
  available: boolean;
  englishTranslation?: string;
  latencyMs?: number;
  model?: string;
  glosses?: string[];
  error?: string;
}

const DEFAULT_SERVER_URL = 'http://127.0.0.1:5005/translate';

export class GlossaBARTClient {
  private serverUrl: string;

  constructor(serverUrl = DEFAULT_SERVER_URL) {
    this.serverUrl = serverUrl;
  }

  async checkHealth(): Promise<boolean> {
    try {
      const res = await fetch('http://127.0.0.1:5005/health', {
        method: 'GET',
        headers: { Accept: 'application/json' },
        signal: AbortSignal.timeout(1500),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  async translate(glosses: string[], useBeamSearch = false): Promise<TranslationResponse> {
    if (glosses.length === 0) {
      return {
        available: true,
        englishTranslation: '',
        latencyMs: 0,
        glosses: [],
      };
    }

    try {
      const res = await fetch(this.serverUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({
          glosses,
          use_beam_search: useBeamSearch,
          num_beams: 4,
        }),
        signal: AbortSignal.timeout(4000),
      });

      if (!res.ok) {
        return {
          available: false,
          error: `Server returned status ${res.status}`,
        };
      }

      const data = await res.json();
      return {
        available: true,
        englishTranslation: data.english_translation,
        latencyMs: data.latency_ms,
        model: data.model,
        glosses,
      };
    } catch {
      return {
        available: false,
        error: 'Translation server offline (port 5005). Running in Gloss Accumulator Mode.',
      };
    }
  }

  speakEnglish(text: string): boolean {
    if (!('speechSynthesis' in window) || !text.trim()) {
      return false;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = 1.0;
    window.speechSynthesis.speak(utterance);
    return true;
  }
}

