import type { ReactNode } from "react";

export const Card = ({
  title,
  className,
  children,
}: {
  title?: string;
  className?: string;
  children: ReactNode;
}) => (
  <div
    className={`rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-4 shadow-[0_1px_2px_rgba(16,24,40,0.04)] ${className ?? ""}`}
  >
    {title && <strong className="mb-2 block text-sm font-semibold text-[var(--ink)]">{title}</strong>}
    <div>{children}</div>
  </div>
);
