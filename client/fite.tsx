import React from 'react';
import './style.css';

const TAF_STRING = [
  `\
TAF KADW 280100Z 0701/0807 01010KT 8000 TSRA BKN030CB QNH2902INS
BECMG 0704/0705 01015G17KT 9999 BKN020 BKN025 QNH2902INS
BECMG 0705/0706 VRB06KT 9999 BKN020 QNH2902INS TX13/0421Z TNM03/0508Z`,
  `\
TAF KADW 280200Z 0701/0807 01010KT 8000 TSRA BKN030CB QNH2902INS
BECMG 0704/0705 01015G17KT 9999 BKN020 BKN025 QNH2902INS
BECMG 0705/0706 VRB06KT 9999 BKN020 QNH2902INS TX13/0421Z TNM03/0508Z`
]
interface FITEState {
  taf: string;
  suggestionsList: string[];
  suggestionIndex: number;
  start: number;
  end: number;
}
type PartialState = Partial<FITEState>;

type FITEActionState = FITEState & {
  suggestion: string;
  setState: (state: PartialState) => void
  handleKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  handleChange: (event: React.ChangeEvent<HTMLTextAreaElement>) => void;
}
function initilazieState(): FITEState {
  return {
    taf: "TAF ",
    suggestionsList: TAF_STRING,
    suggestionIndex: 0,
    start: 0,
    end: 0
  }
}


function useFITE(): FITEActionState {

  const [state, dispatchState] = React.useState<FITEState>(initilazieState);
  const { taf, start, end, suggestionsList, suggestionIndex } = state;
  const setState = React.useCallback((state: PartialState) => dispatchState(prevState => ({ ...prevState, ...state })), [dispatchState]);
  const suggestion = React.useMemo(() => state.suggestionsList[state.suggestionIndex], [state.suggestionsList, state.suggestionIndex]);

  const handleKeyDown = React.useCallback((event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    const { key, ctrlKey } = event;
    console.log(event);
    if (ctrlKey && key === 'ArrowUp') {
      event.preventDefault();
      setState({ suggestionIndex: (suggestionIndex + 1) % suggestionsList.length });
    }
    else if (ctrlKey && key === 'ArrowDown') {
      event.preventDefault();
      setState({ suggestionIndex: (suggestionIndex + suggestionsList.length - 1) % suggestionsList.length });
    }
    else if (key === 'Tab') {
      event.preventDefault();
      setState({ taf: suggestionsList[suggestionIndex] });
    }
  }, [suggestionsList, suggestionIndex]);

  const handleChange = React.useCallback((event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const { value } = event.target;
    setState({ taf: value, start: value.length, end: value.length });
  }, []);


  return { ...state, suggestion, setState, handleKeyDown, handleChange };

}





function FITE() {
  /* a text area with intelisense autocomplete
      that is used for writing a terminal aerodrome forecast
      the intelisense is based the TAF_STRING above
      the user can cycle through the intelisense options with the up and down arrow keys
      the user can select an option by pressing tab or enter */

  const {
    taf,
    start,
    end,
    suggestion,
    suggestionsList,
    suggestionIndex,
    setState,
    handleKeyDown,
    handleChange
    // ...state
  } = useFITE();

  return (
    <div className='taf-component-container'>
      <div style={{ position: 'absolute' }}>
        <pre className='taf-annotation-layer'>{suggestion}</pre>
      </div>
      <textarea className='taf-input-layer' onKeyDown={handleKeyDown} onChange={handleChange} value={taf} />
    </div>
  );
}


export default FITE;
