export const Input = ({
  value,
  onChange,
  className = "",
  ...props
}: {
  value: string | number;
  onChange: (value: string | number) => void;
  className?: string;
} & React.InputHTMLAttributes<HTMLInputElement>) => {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`
        bg-gray-700
        w-full
        border
        border-gray-600
        text-gray-300
        rounded
        p-2
        outline-none
        focus:border-gray-500
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
