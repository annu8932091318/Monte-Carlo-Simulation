import { ChevronDown } from "lucide-react";
import React from "react";
import { cn } from "../../lib/utils";

const Select = React.forwardRef(({ className, children, value, onValueChange, ...props }, ref) => {
    return (
        <div className="relative">
            <select
                className={cn(
                    "flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
                    className
                )}
                ref={ref}
                value={value}
                onChange={(e) => onValueChange && onValueChange(e.target.value)}
                {...props}
            >
                {children}
            </select>
            <ChevronDown className="absolute right-3 top-3 h-4 w-4 opacity-50 pointer-events-none" />
        </div>
    );
});
Select.displayName = "Select";

const SelectItem = React.forwardRef(({ className, children, value, ...props }, ref) => {
    return (
        <option
            className={cn("relative flex cursor-default select-none items-center py-1.5 pl-8 pr-2 text-sm outline-none", className)}
            ref={ref}
            value={value}
            {...props}
        >
            {children}
        </option>
    );
});
SelectItem.displayName = "SelectItem";

export { Select, SelectItem };
