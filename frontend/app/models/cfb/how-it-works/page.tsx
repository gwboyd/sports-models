import type { Metadata } from "next";
import { ModelMethodology } from "@/app/components/ModelMethodology";
import { getModelInfoMarkdown } from "@/app/lib/model-info";

export const metadata: Metadata = { title: "How the CFB Model Works" };

export default async function HowItWorksPage() {
  const content = await getModelInfoMarkdown("cfb");
  return <ModelMethodology content={content} />;
}
