"use client";

import type { InputHTMLAttributes } from "react";

export const Input = ({
  value,
  onChange,
  className = "",
  ...props
}: {
  value: string | number;
  onChange: (value: string | number) => void;
  className?: string;
} & InputHTMLAttributes<HTMLInputElement>) => {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`
        bg-white
        w-full
        border
        border-[var(--border)]
        text-[var(--ink)]
        rounded-xl
        px-3
        py-2.5
        min-h-11
        outline-none
        focus:border-[var(--accent)]
        focus:ring-4
        focus:ring-blue-100
        transition-colors
        duration-100
        [&::-webkit-outer-spin-button]:appearance-none
        [&::-webkit-outer-spin-button]:m-0
        [&::-webkit-inner-spin-button]:appearance-none
        [&::-webkit-inner-spin-button]:m-0
        [&[type='number']]:appearance-textfield
        ${className}
      `}
      {...props}
    />
  );
};
