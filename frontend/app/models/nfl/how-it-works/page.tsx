/* eslint-disable @next/next/no-img-element */
import type { Metadata } from "next";
import Markdown from "react-markdown";
import { getModelInfoMarkdown } from "@/app/lib/model-info";

export const metadata: Metadata = { title: "How the NFL Model Works" };

export default async function HowItWorksPage() {
  const content = await getModelInfoMarkdown();
  return (
    <main className="mx-auto w-full max-w-4xl px-4 py-8 pb-20 sm:px-6 sm:py-10 lg:px-8">
      <article className="rounded-lg border border-[var(--border)] bg-white px-5 py-6 sm:px-9 sm:py-8">
        <Markdown components={{
          h1: ({ children }) => <h1 className="text-3xl font-bold tracking-tight text-[var(--ink)] sm:text-4xl">{children}</h1>,
          h2: ({ children }) => <h2 className="mt-10 border-t border-[var(--border)] pt-8 text-2xl font-semibold tracking-tight text-[var(--ink)]">{children}</h2>,
          h3: ({ children }) => <h3 className="mt-7 text-xl font-semibold tracking-tight text-[var(--ink)]">{children}</h3>,
          p: ({ children }) => <p className="mt-4 text-[15px] leading-7 text-[var(--muted)] sm:text-base">{children}</p>,
          ul: ({ children }) => <ul className="mt-4 list-disc space-y-2 pl-5 text-[15px] leading-7 text-[var(--muted)] sm:text-base">{children}</ul>,
          ol: ({ children }) => <ol className="mt-4 list-decimal space-y-2 pl-5 text-[15px] leading-7 text-[var(--muted)] sm:text-base">{children}</ol>,
          li: ({ children }) => <li className="pl-1">{children}</li>,
          strong: ({ children }) => <strong className="font-semibold text-[var(--ink)]">{children}</strong>,
          em: ({ children }) => <em className="text-[var(--ink)]">{children}</em>,
          a: ({ href, children }) => <a href={href} className="font-medium text-[var(--accent)] underline decoration-blue-200 underline-offset-4 hover:decoration-[var(--accent)]">{children}</a>,
          code: ({ children }) => <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[0.9em] text-[var(--ink)]">{children}</code>,
          img: ({ src, alt }) => <img src={typeof src === "string" ? src : undefined} alt={alt ?? ""} className="mt-5 h-auto w-full rounded-md border border-[var(--border)] bg-white" loading="lazy" />,
        }}>{content}</Markdown>
      </article>
    </main>
  );
}
