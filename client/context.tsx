import React from "react";

export interface FITEState {
  apiUrl: string;
  model: string;
  strategy: string;
  textPrompt?: string;
  textCompletion?: string;
  textAreaValue?: string;
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

export default ({ children, ...initialState }: React.PropsWithChildren<FITEState>) => {
  const [state, __setState] = React.useState({ ...initialState });

  return <FITEContext.Provider value={{ ...state, __setState }}>{state.apiUrl && children}</FITEContext.Provider>;
};
