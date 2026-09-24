import express from "express";
import path from "path";
import fs from "fs";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type, Schema } from "@google/genai";

const PORT = 3000;

function resolveApiKey(): string {
  for (const envFile of [".env.local", ".env"]) {
    try {
      if (fs.existsSync(envFile)) {
        const content = fs.readFileSync(envFile, "utf-8");
        const match = content.match(/^GEMINI_API_KEY\s*=\s*(.+)$/m);
        if (match && match[1]?.trim()) {
          const val = match[1].trim().replace(/^["']|["']$/g, "");
          if (val) return val;
        }
      }
    } catch {
      // ignore
    }
  }
  return (
    process.env.GEMINI_API_KEY ||
    process.env.API_KEY ||
    ""
  );
}

function getGeminiClient(): GoogleGenAI {
  const apiKey = resolveApiKey();
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY environment variable is not configured.");
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

const isQuotaOrRateLimitOrUnavailableError = (error: any): boolean => {
  const msg = (error?.message || error?.toString?.() || "").toLowerCase();
  const status = error?.status || error?.statusCode || error?.code;
  return (
    status === 429 ||
    status === 503 ||
    msg.includes("429") ||
    msg.includes("503") ||
    msg.includes("unavailable") ||
    msg.includes("high demand") ||
    msg.includes("quota") ||
    msg.includes("rate limit") ||
    msg.includes("resource_exhausted") ||
    msg.includes("resourceexhausted") ||
    msg.includes("exceeded")
  );
};

const MODELS_TO_TRY = ["gemini-3.8-flash", "gemini-3.1-flash-lite"];

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function executeWithFallback<T>(
  executor: (ai: GoogleGenAI, modelName: string) => Promise<T>
): Promise<T> {
  const ai = getGeminiClient();
  let lastError: any = null;

  for (const model of MODELS_TO_TRY) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        return await executor(ai, model);
      } catch (error: any) {
        lastError = error;
        if (isQuotaOrRateLimitOrUnavailableError(error)) {
          if (attempt === 0) {
            await sleep(1500);
            continue;
          }
          console.warn(`Model ${model} hit quota/availability limit. Falling back to alternative model...`);
          break;
        }
        // If it's a structural or validation error, rethrow immediately
        throw error;
      }
    }
  }

  // If all models failed due to quota/rate limit:
  if (isQuotaOrRateLimitOrUnavailableError(lastError)) {
    throw new Error(
      "Gemini API Quota or Rate Limit temporarily reached across models. Please wait 30 seconds and retry."
    );
  }

  throw lastError || new Error("Failed to generate content with Gemini.");
}

const resumeSchema: Schema = {
  type: Type.OBJECT,
  properties: {
    summary: { type: Type.STRING, description: "Professional summary tailored to the role" },
    skills: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of relevant technical skills" },
    certifications: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of relevant certifications" },
    experience: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          company: { type: Type.STRING },
          role: { type: Type.STRING },
          duration: { type: Type.STRING },
          bullets: { type: Type.ARRAY, items: { type: Type.STRING } }
        },
        required: ["company", "role", "duration", "bullets"]
      }
    },
    analysis: {
      type: Type.OBJECT,
      properties: {
        matchScore: { type: Type.NUMBER, description: "Score from 0-100 indicating fit" },
        keywordsUsed: { type: Type.ARRAY, items: { type: Type.STRING } },
        toneNotes: { type: Type.STRING, description: "Explanation of tone adjustments" }
      },
      required: ["matchScore", "keywordsUsed", "toneNotes"]
    }
  },
  required: ["summary", "skills", "certifications", "experience", "analysis"]
};

const answerSchema: Schema = {
  type: Type.OBJECT,
  properties: {
    generated_answer: { type: Type.STRING },
    confidence_note: { type: Type.STRING },
    intent_detected: { type: Type.STRING, description: "Technical, Behavioral, Governance, or Role Fit" }
  },
  required: ["generated_answer", "confidence_note", "intent_detected"]
};

const coverLetterSchema: Schema = {
  type: Type.OBJECT,
  properties: {
    content: { type: Type.STRING, description: "The full markdown formatted cover letter" },
  },
  required: ["content"]
};

const parsedResumeSchema: Schema = {
  type: Type.OBJECT,
  properties: {
    name: { type: Type.STRING },
    contact: {
      type: Type.OBJECT,
      properties: {
        location: { type: Type.STRING },
        email: { type: Type.STRING },
        phone: { type: Type.STRING },
        linkedin: { type: Type.STRING },
      },
      required: ["location", "email"]
    },
    summary: { type: Type.STRING },
    skills: { type: Type.ARRAY, items: { type: Type.STRING } },
    certifications: { type: Type.ARRAY, items: { type: Type.STRING } },
    education: { type: Type.ARRAY, items: { type: Type.STRING } },
    experience: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          company: { type: Type.STRING },
          role: { type: Type.STRING },
          duration: { type: Type.STRING },
          bullets: { type: Type.ARRAY, items: { type: Type.STRING } }
        },
        required: ["company", "role", "duration", "bullets"]
      }
    }
  },
  required: ["name", "contact", "summary", "skills", "experience"]
};

async function startServer() {
  const app = express();
  app.use(express.json({ limit: "10mb" }));

  // API Health Check
  app.get("/api/health", (_req, res) => {
    res.json({
      status: "ok",
      hasKey: Boolean(process.env.GEMINI_API_KEY || process.env.API_KEY),
    });
  });

  // Tailored Resume Generation
  app.post("/api/tailor-resume", async (req, res) => {
    try {
      const { jobDescription, targetRole, baseResume, jobLink } = req.body;
      if (!jobDescription || !targetRole || !baseResume) {
        return res.status(400).json({ error: "Missing required fields: jobDescription, targetRole, baseResume" });
      }

      const systemPrompt = `
        You are RoleArchitect, a sophisticated career strategist engine. 
        Your goal is to rewrite the candidate's experience to perfectly align with a specific Job Description (JD) and Role.
        
        TARGET ROLE: ${targetRole}

        CORE RULES:
        1. FACTUAL INTEGRITY: Do not invent experiences. Reframe and elaborate on existing facts using the JD's terminology.
        2. TECHNICAL DEPTH & CONTEXT: 
           - Maximize technical detail. Never simplify. 
           - Every bullet point must include specific tools, protocols, versions, or methodologies.
           - Context is king: Explain *why* a task was done, the *complexity* involved, and the *architectural impact*.
        3. QUANTITY & LENGTH: 
           - DO NOT remove experience. Keep all roles.
           - Generate AT LEAST 8 dense bullet points per role (ideally 10-12 for Senior roles).
           - The final output should be comprehensive and verbose enough to fill 2+ pages.
        4. TONE: Professional, authoritative, highly technical. Use "Senior/Architect" level language.
        5. ROLE INTELLIGENCE:
           - If Cloud Security: Focus on risk, governance, audit, IAM, WAF, Zero Trust, Compliance frameworks (NIST, SOC2).
           - If Cloud Engineer (General): Focus on reliability, scale, cost optimization, IaC patterns, multi-region architectures.
           - If DevSecOps: Focus on CI/CD security, container hardening, shift-left security, policy-as-code.
           - If IAM Engineer: Focus on Identity Lifecycle (JML), Access Governance, AuthN/AuthZ (SAML/OIDC/OAuth), PAM, and Federation.
           - If DevOps Engineer: Focus on CI/CD pipeline design, IaC, system reliability, operational maturity, and observability.
           - If AWS Cloud Engineer: Focus on AWS core services (EC2, VPC, S3, RDS), infrastructure design, high availability, and cost optimization.
           - If Azure Cloud Engineer: Focus on Azure compute, networking, storage, VNets, high availability, and disaster recovery.
           - If Cloud Solution Architect: Focus on end-to-end solution design, cloud patterns, multi-tier systems, NFRs, and cost modeling.
           - If Azure Cloud Architect: Focus on Azure enterprise architecture, landing zones, management groups, governance, and platform design.
           - If Site Reliability Engineer: Focus on service reliability, uptime, SLIs/SLOs, error budgets, incident response, and observability.
           - If Information Security Governance: Focus on governance frameworks, policy lifecycle management, risk appetite, and board-level reporting.
           - If GRC Operations / Risk Management: Focus on integrated GRC language, control framework alignment (NIST, ISO, SOC 2), risk register governance, and cross-functional coordination.
           - If Cybersecurity Risk Management: Focus on inherent and residual risk assessments, threat modeling, vulnerability correlation, and control effectiveness evaluation.
           - If Senior Cybersecurity Risk Management: Focus on enterprise-level technical risk assessment, threat modeling, risk quantification, and control effectiveness.
           - If Cybersecurity Risk Control: Focus on control design, validation, effectiveness language, risk-control mapping, and remediation tracking.
           - If Information Security Compliance Risk Management: Focus on compliance execution, regulatory alignment language, compliance risk identification, and tracking.
           - If Information Security Risk and Compliance: Focus on balanced risk and compliance execution language, control mapping, and gap analysis.
           - If Compliance & Regulatory Affairs: Focus on regulatory mapping (NIST, ISO, GDPR), audit preparation, and documentation.
           - If Compliance Analyst: Focus on compliance monitoring, audit support language, policy adherence, and documentation management.
           - If Privacy & Data Protection: Focus on data protection laws, privacy impact assessments, and data lifecycle governance.
           - If Third-Party / Vendor Risk Management: Focus on vendor assessments, due diligence, and contract risk evaluation.
           - If Senior IT Auditor: Focus on independent assurance, IT audit planning and execution, control design and effectiveness testing, and audit findings reporting.
           - If IT Audit & Controls Assurance: Focus on control testing, audit execution, and evidence validation.
           - If IAM Governance, Risk & Compliance: Focus on access reviews, segregation of duties (SoD), and entitlement governance.
           - If AML & Fraud Investigator: Focus on integrated AML, fraud, and financial crime investigations, prioritizing transaction monitoring, KYC/CDD/EDD reviews, customer risk assessments, SAR preparation, and sanctions screening.

        INPUT DATA:
        ${JSON.stringify(baseResume)}
      `;

      const userPrompt = `
        JOB CONTEXT:
        ${jobLink ? `Job Link: ${jobLink}` : ""}
        
        JOB DESCRIPTION:
        ${jobDescription}

        INSTRUCTIONS:
        1. Analyze the JD for key technical requirements and soft skills.
        2. Rewrite the "Summary" to be a comprehensive, technical executive summary (4-6 sentences).
        3. Reconstruct the "Experience" bullets:
           - EXPAND on the base resume's points. Do not summarize.
           - Ensure a minimum of 8 high-quality, dense bullet points per job role.
           - For every point, strictly follow: Action Verb -> Deep Technical Context -> Specific Tools Used -> Quantitative Business Impact.
           - Make all conversions extremely technical.
        4. Curate the "Skills" and "Certifications" sections. Ensure relevant certifications from the input are included.
        5. Return JSON only conforming to the schema.
      `;

      const tailoredData = await executeWithFallback(async (ai, model) => {
        const response = await ai.models.generateContent({
          model,
          contents: [
            { role: "user", parts: [{ text: systemPrompt + "\n\n" + userPrompt }] }
          ],
          config: {
            responseMimeType: "application/json",
            responseSchema: resumeSchema,
            temperature: 0.4
          }
        });

        if (response.text) {
          return JSON.parse(response.text);
        }
        throw new Error("No response generated by Gemini model.");
      });

      return res.json(tailoredData);
    } catch (error: any) {
      console.error("Tailor Resume Error:", error);
      const statusCode = isQuotaOrRateLimitOrUnavailableError(error) ? 429 : 500;
      return res.status(statusCode).json({ error: error.message || "Failed to generate tailored resume" });
    }
  });

  // Optimize Tailored Resume
  app.post("/api/optimize-resume", async (req, res) => {
    try {
      const { currentResume, userPrompt, jobDescription, targetRole } = req.body;
      if (!currentResume || !userPrompt) {
        return res.status(400).json({ error: "Missing required fields: currentResume, userPrompt" });
      }

      const systemPrompt = `
        You are RoleArchitect, a sophisticated career strategist engine. 
        Your goal is to update and optimize the candidate's tailored resume based on the user's specific request.
        
        TARGET ROLE: ${targetRole || ""}
        
        CURRENT RESUME JSON:
        ${JSON.stringify(currentResume)}
        
        JOB DESCRIPTION:
        ${jobDescription || ""}

        USER REQUEST:
        ${userPrompt}

        INSTRUCTIONS:
        1. Modify the CURRENT RESUME JSON to fulfill the USER REQUEST.
        2. Maintain the factual integrity of the resume. Do not invent experiences unless asked to rephrase existing ones.
        3. Ensure the output strictly follows the JSON schema.
        4. Update the "analysis.toneNotes" to briefly explain what you changed based on the user's request.
        5. Return JSON only conforming to the schema.
      `;

      const updated = await executeWithFallback(async (ai, model) => {
        const response = await ai.models.generateContent({
          model,
          contents: [
            { role: "user", parts: [{ text: systemPrompt }] }
          ],
          config: {
            responseMimeType: "application/json",
            responseSchema: resumeSchema,
            temperature: 0.4
          }
        });

        if (response.text) {
          return JSON.parse(response.text);
        }
        throw new Error("No response generated by Gemini model.");
      });

      return res.json(updated);
    } catch (error: any) {
      console.error("Optimize Resume Error:", error);
      const statusCode = isQuotaOrRateLimitOrUnavailableError(error) ? 429 : 500;
      return res.status(statusCode).json({ error: error.message || "Failed to optimize resume" });
    }
  });

  // Application Q&A
  app.post("/api/application-answer", async (req, res) => {
    try {
      const { question, targetRole, baseProfile, wordLimit, jobDescription, jobLink } = req.body;
      if (!question || !baseProfile) {
        return res.status(400).json({ error: "Missing required fields: question, baseProfile" });
      }

      const systemPrompt = `
        You are an intelligent Application Question Assistant.
        Your task is to answer employer application questions based on a candidate's profile.

        TARGET ROLE: ${targetRole || ""}
        WORD LIMIT: ${wordLimit ? wordLimit + " words" : "Concise (approx 200 words)"}

        QUESTION INTENT CLASSIFICATION RULES:
        1. Technical (Tools, systems): Answer with experience-driven, factual depth.
        2. Behavioral ("Describe a time..."): Use STAR-aligned but concise structure.
        3. Governance (Risk, compliance): Focus on control, audit, and rigor.
        4. Role Fit ("Why this role?"): Focus on alignment and competence, NOT enthusiasm or marketing fluff.

        TONE SUPPRESSION RULES:
        - NO overly polished transitions ("Furthermore", "Moreover").
        - NO marketing language ("Thrilled", "Excited", "Passionate").
        - NO generic claims.
        - Style: Neutral, evidence-based, professionally understated.
        - Format: Plain text, no bullets, no markdown.

        CANDIDATE PROFILE:
        ${JSON.stringify(baseProfile)}

        JOB CONTEXT:
        ${jobLink ? `Job Link: ${jobLink}` : ""}
        ${jobDescription ? `Job Description:\n${jobDescription}` : ""}
      `;

      const userPromptText = `
        APPLICATION QUESTION:
        "${question}"

        Generate a tailored answer following the rules above.
      `;

      const answerResult = await executeWithFallback(async (ai, model) => {
        const response = await ai.models.generateContent({
          model,
          contents: [{ role: "user", parts: [{ text: systemPrompt + "\n\n" + userPromptText }] }],
          config: {
            responseMimeType: "application/json",
            responseSchema: answerSchema,
            temperature: 0.3
          }
        });

        if (response.text) {
          return JSON.parse(response.text);
        }
        throw new Error("No response generated by Gemini model.");
      });

      return res.json(answerResult);
    } catch (error: any) {
      console.error("Application Answer Error:", error);
      const statusCode = isQuotaOrRateLimitOrUnavailableError(error) ? 429 : 500;
      return res.status(statusCode).json({ error: error.message || "Failed to generate answer" });
    }
  });

  // Cover Letter Generation
  app.post("/api/cover-letter", async (req, res) => {
    try {
      const { companyName, hiringManager, targetRole, baseResume, jobDescription } = req.body;
      if (!companyName || !baseResume) {
        return res.status(400).json({ error: "Missing required fields: companyName, baseResume" });
      }

      const systemPrompt = `
        You are an Executive Career Strategist.
        Write a highly tailored Cover Letter.

        TARGET ROLE: ${targetRole || ""}
        COMPANY: ${companyName}
        HIRING MANAGER: ${hiringManager || "Hiring Manager"}

        TONE & STYLE:
        - Confidence without arrogance.
        - Evidence-based statements (cite specific technical wins from the resume).
        - "Hook" opening that addresses the company's specific needs found in the JD.
        - No generic fluff ("I am a hard worker").
        - Use role-specific vocabulary aligned with ${targetRole || "the targeted position"}.

        STRUCTURE:
        1. Header (Standard business format)
        2. Salutation
        3. The Hook: Why this specific role + company? Connect to a specific JD requirement.
        4. The Value Prop: 2 specific technical achievements from the resume that prove you can solve their problems.
        5. The Closing: Professional call to action.

        CANDIDATE DATA:
        ${JSON.stringify(baseResume)}

        JOB DESCRIPTION:
        ${jobDescription || "No specific JD provided, focus on general role excellence."}
      `;

      const coverLetterResult = await executeWithFallback(async (ai, model) => {
        const response = await ai.models.generateContent({
          model,
          contents: [{ role: "user", parts: [{ text: systemPrompt }] }],
          config: {
            responseMimeType: "application/json",
            responseSchema: coverLetterSchema,
            temperature: 0.4
          }
        });

        if (response.text) {
          return JSON.parse(response.text);
        }
        throw new Error("No response generated by Gemini model.");
      });

      return res.json(coverLetterResult);
    } catch (error: any) {
      console.error("Cover Letter Error:", error);
      const statusCode = isQuotaOrRateLimitOrUnavailableError(error) ? 429 : 500;
      return res.status(statusCode).json({ error: error.message || "Failed to generate cover letter" });
    }
  });

  // Parse Resume from Text / Uploaded Document
  app.post("/api/parse-resume", async (req, res) => {
    try {
      const { text } = req.body;
      if (!text) {
        return res.status(400).json({ error: "Missing required field: text" });
      }

      const prompt = `
        Extract structured resume data from the text below. 
        Map it to the JSON schema strictly.
        Ensure "bullets" in experience are preserved as individual points from the source text.
        
        RESUME TEXT:
        ${text}
      `;

      const parsedData = await executeWithFallback(async (ai, model) => {
        const response = await ai.models.generateContent({
          model,
          contents: [{ role: "user", parts: [{ text: prompt }] }],
          config: {
            responseMimeType: "application/json",
            responseSchema: parsedResumeSchema
          }
        });

        if (response.text) {
          return JSON.parse(response.text);
        }
        throw new Error("No response generated by Gemini model.");
      });

      return res.json(parsedData);
    } catch (error: any) {
      console.error("Parse Resume Error:", error);
      const statusCode = isQuotaOrRateLimitOrUnavailableError(error) ? 429 : 500;
      return res.status(statusCode).json({ error: error.message || "Failed to parse resume" });
    }
  });

  // Vite middleware for development vs static production serve
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    // Express 5 wildcard route
    app.get("*all", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`RoleArchitect server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
