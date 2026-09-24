import { ResumeData, TargetRole, TailoredResume } from '../types';

export interface ApplicationAnswerResponse {
  generated_answer: string;
  confidence_note: string;
  intent_detected: string;
}

export interface CoverLetterResponse {
  content: string;
}

async function handleApiResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorMsg = `Server error (${res.status})`;
    try {
      const data = await res.json();
      if (data?.error) {
        errorMsg = data.error;
      }
    } catch {
      // Non-JSON error
      const text = await res.text().catch(() => '');
      if (text) errorMsg = text;
    }
    throw new Error(errorMsg);
  }
  return res.json() as Promise<T>;
}

export const generateTailoredResume = async (
  jobDescription: string,
  targetRole: TargetRole,
  baseResume: ResumeData,
  jobLink?: string
): Promise<TailoredResume> => {
  const res = await fetch('/api/tailor-resume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      jobDescription,
      targetRole,
      baseResume,
      jobLink
    })
  });
  return handleApiResponse<TailoredResume>(res);
};

export const optimizeTailoredResume = async (
  currentResume: TailoredResume,
  userPrompt: string,
  jobDescription: string,
  targetRole: TargetRole
): Promise<TailoredResume> => {
  const res = await fetch('/api/optimize-resume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      currentResume,
      userPrompt,
      jobDescription,
      targetRole
    })
  });
  return handleApiResponse<TailoredResume>(res);
};

export const generateApplicationAnswer = async (
  question: string,
  targetRole: TargetRole,
  baseProfile: ResumeData,
  wordLimit?: number,
  jobDescription?: string,
  jobLink?: string
): Promise<ApplicationAnswerResponse> => {
  const res = await fetch('/api/application-answer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      targetRole,
      baseProfile,
      wordLimit,
      jobDescription,
      jobLink
    })
  });
  return handleApiResponse<ApplicationAnswerResponse>(res);
};

export const generateCoverLetter = async (
  companyName: string,
  hiringManager: string,
  targetRole: TargetRole,
  baseResume: ResumeData,
  jobDescription?: string
): Promise<CoverLetterResponse> => {
  const res = await fetch('/api/cover-letter', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      companyName,
      hiringManager,
      targetRole,
      baseResume,
      jobDescription
    })
  });
  return handleApiResponse<CoverLetterResponse>(res);
};

export const parseResumeFromText = async (text: string): Promise<ResumeData> => {
  const res = await fetch('/api/parse-resume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  return handleApiResponse<ResumeData>(res);
};
