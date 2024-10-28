export const SectionTitle = ({
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) => (
  <h4 className="text-gray-300 text-xl font-bold" {...props} />
);

export const BodyText = ({
  ...props
}: React.HTMLAttributes<HTMLParagraphElement>) => (
  <p className="text-gray-400 mb-1" {...props} />
);
