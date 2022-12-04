import React from "react";
interface FiteProps {
  apiUrl: string | URL;
  // children: React.ReactNode;
  model: string;
  strategy: string;
  textAreaValue?: string;
}
export interface FITEState extends FiteProps {
  // apiUrl: URL;
  models?: string[];
  strategies?: string[];
  textPrompt?: string;
  textCompletion?: string;
}
interface ContextState extends FITEState {
  __setState: (ctx: React.SetStateAction<FITEState>) => void;
}

export const FITEContext = React.createContext<ContextState>({
  apiUrl: "http://localhost:8000",
  model: "gpt2-taf-base1",
  strategy: "GREEDY",
  textAreaValue: "",
  __setState: () => void 0,
});

export default ({ children, ...initialState }: React.PropsWithChildren<FiteProps>) => {
  const [state, __setState] = React.useState<FiteProps>(initialState); // as [FITEState, React.Dispatch<React.SetStateAction<FITEState>>];

  return <FITEContext.Provider value={{ ...state, __setState }}>{children}</FITEContext.Provider>;
};
