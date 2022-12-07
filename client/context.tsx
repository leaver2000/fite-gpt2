import React from "react";

export const TAF_HEADER = "TAF ";
export const TAF_HEADER_LENGTH = TAF_HEADER.length;

interface FITEProps {
  apiUrl: string | URL;
  model: string;
  strategy: string;
}

export interface FITEState extends FITEProps {
  modelOptions?: string[];
  strategyOptions?: string[];
  textAreaValue: string;
  textCompletion: string;
  textPrompt?: string;
}

const DEFAULT_STATE = {
  apiUrl: "http://localhost:8000",
  model: "gpt2-taf-base1",
  strategy: "GREEDY",
  textAreaValue: TAF_HEADER,
  textCompletion: TAF_HEADER,
} as FITEState;

interface ContextState extends FITEState {
  __setState: (ctx: React.SetStateAction<FITEState>) => void;
}

export const FITEContext = React.createContext<ContextState>({
  ...DEFAULT_STATE,
  __setState: () => void 0,
});

export default ({ children, ...initialState }: React.PropsWithChildren<FITEProps>) => {
  const [state, __setState] = React.useState<FITEState>({ ...DEFAULT_STATE, ...initialState });

  return <FITEContext.Provider value={{ ...state, __setState }}>{children}</FITEContext.Provider>;
};
