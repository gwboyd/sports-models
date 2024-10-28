export const Card = ({
  title,
  className,
  children,
}: {
  title?: string;
  className?: string;
  children: React.ReactNode;
}) => (
  <div
    className={`border border-gray-700 bg-gray-800 rounded p-3 flex flex-col gap-2 ${className}`}
  >
    {title && <strong className="text-gray-300">{title}</strong>}
    <div>{children}</div>
  </div>
);
