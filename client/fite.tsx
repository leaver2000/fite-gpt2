import React from "react";
import "./style.css";

const TAF_STRING = [
  `\
TAF KBLV 280100Z 2801/2807 01012G18KT 9999 BKN030 QNH2902INS
BECMG 0704/0705 01015G17KT 9999 BKN020 BKN025 QNH2902INS
BECMG 0705/0706 VRB06KT 9999 BKN020 QNH2902INS TX13/0421Z TNM03/0508Z`,
  `\
TAF KBLV 280100Z 2801/2807 01015G20KT 8000 TSRA BKN030CB QNH2902INS
BECMG 0704/0705 01015G17KT 9999 BKN020 BKN025 QNH2902INS
BECMG 0705/0706 VRB06KT 9999 BKN020 QNH2902INS TX13/0421Z TNM03/0508Z`,
];
interface FITEState {
  text: string;
  suggestionsList: string[];
  suggestionIndex: number;
  start: number;
  end: number;
}
type PartialState = Partial<FITEState>;

type FITEActionState = FITEState & {
  suggestion: string;
  setState: (state: PartialState) => void;
  handleKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  handleChange: (event: React.ChangeEvent<HTMLTextAreaElement>) => void;
};
function initializeState(): FITEState {
  return {
    text: "TAF KBLV 280100Z 2801/2807 01012G18KT",
    suggestionsList: TAF_STRING,
    suggestionIndex: 0,
    start: 0,
    end: 0,
  };
}

function useFITE(url: string): FITEActionState {
  const [state, dispatchState] = React.useState<FITEState>(initializeState);
  const { text, start, end, suggestionsList, suggestionIndex } = state;
  const setState = React.useCallback(
    (state: PartialState) =>
      dispatchState((prevState) => ({ ...prevState, ...state })),
    [dispatchState]
  );
  const suggestion = React.useMemo(
    () => suggestionsList[suggestionIndex],
    [suggestionsList, suggestionIndex]
  );

  React.useEffect(() => {
    const options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    };
    if (text.endsWith(" ")) {
      fetch(url, options)
        .then((response) => response.json())
        .then((generatedText: string[][]) =>
          setState({ suggestionsList: [generatedText.join("\n")] })
        );
    }
  }, [url, text, setState]);

  const handleKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      const { key, ctrlKey } = event;
      const { selectionStart, selectionEnd } =
        event.target as HTMLTextAreaElement;

      if (ctrlKey && key === "ArrowUp") {
        event.preventDefault();
        setState({
          suggestionIndex: (suggestionIndex + 1) % suggestionsList.length,
        });
      } else if (ctrlKey && key === "ArrowDown") {
        event.preventDefault();
        setState({
          suggestionIndex:
            (suggestionIndex + suggestionsList.length - 1) %
            suggestionsList.length,
        });
      } else if (ctrlKey && key === "Enter") {
        event.preventDefault();
        setState({ text: suggestionsList[suggestionIndex] });
      } else if (key === "Tab") {
        event.preventDefault();
        // update the taf value with the next word in the suggestion
        const nextWord = suggestion.split(" ")[start];
        const newText = text.slice(0, end) + nextWord + text.slice(end);
        setState({ text: newText });
      }
      // update the start and end values
      // setState({ start: selectionStart, end: selectionEnd });
    },
    [suggestionsList, suggestionIndex, suggestion, text, start, end, setState]
  );

  const handleChange = React.useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      const { value } = event.target;
      setState({ text: value, start: value.length, end: value.length });
    },
    []
  );

  return { ...state, suggestion, setState, handleKeyDown, handleChange };
}

function FITE({ apiUrl, ...props }: { apiUrl: string }) {
  /* a text area with intellisense autocomplete
      that is used for writing a terminal aerodrome forecast
      the intellisense is based the TAF_STRING above
      the user can cycle through the intellisense options with the up and down arrow keys
      the user can select an option by pressing tab or enter */

  const {
    // taf,
    start,
    end,
    suggestion,
    suggestionsList,
    suggestionIndex,
    setState,
    handleKeyDown,
    handleChange,
    // ...state
  } = useFITE(apiUrl);

  return (
    <div className="taf-component-container">
      <div style={{ position: "absolute" }}>
        <pre className="taf-annotation-layer">{suggestion}</pre>
      </div>
      <textarea
        className="taf-input-layer"
        onKeyDown={handleKeyDown}
        onChange={handleChange}
        // value={taf}
      />
    </div>
  );
}

export default FITE;
