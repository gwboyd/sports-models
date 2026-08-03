import type { HTMLAttributes } from "react";

export const SectionTitle = ({
  ...props
}: HTMLAttributes<HTMLHeadingElement>) => (
  <h2 className="text-xl font-semibold tracking-tight text-[var(--ink)] sm:text-2xl" {...props} />
);

export const BodyText = ({
  ...props
}: HTMLAttributes<HTMLParagraphElement>) => (
  <p className="mb-1 text-sm leading-6 text-[var(--muted)]" {...props} />
);
