import React from "react";
import ReactDOM from "react-dom";
import FITE from "./fite";
interface SelectionState {
  model: null | string;
  strategy: null | string;
}
const defaultState: SelectionState = {
  model: null,
  strategy: null,
};

function withFetch<T>(url: string, options: RequestInit, callback: (data: T) => void) {
  fetch(url, options)
    .then((response) => response.json())
    .then(callback);
}



function App() {
  const apiUrl = "http://localhost:8000";
  const options = { method: "GET" };
  // fetch the list of models from the api
  const [models, setModels] = React.useState<string[]>([]);
  const [strategies, setStrategy] = React.useState<string[]>([]);
  // state for radio buttons with type script type inference
  const [{ model, strategy }, setSelection] = React.useState<SelectionState>(defaultState);
  
  // fetch the list of models and strategies from the api
  React.useEffect(() => {
    withFetch(`${apiUrl}/models/`, options, setModels);
    withFetch(`${apiUrl}/strategies/`, options, setStrategy);
  }, []);

  const handleModelChange = React.useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const { value:model } = event.target;
      setSelection(({ strategy }) => ({ strategy, model }));
    },
    []
  );

  const handleStrategyChange = React.useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const { value:strategy } = event.target;
      setSelection(({ model }) => ({ model, strategy }));
    },
    []
  );
  //  set default values
  React.useEffect(() => {
    if (!model && models.length > 0) {
      setSelection(({ strategy }) => ({ strategy, model: models[0] }));
    }
    if (!strategy && strategies.length > 0) {
      setSelection(({ model }) => ({ model, strategy: strategies[0] }));
    }
  }, [model, models, strategy, strategies]);

  return (
    <div>
      {/* for each model add a radio button */}
      {models &&
        models.map((model) => (
          <div key={model}>
            <input
              type="radio"
              name="model"
              value={model}
              onChange={handleModelChange}
            />
            <label>{model}</label>
          </div>
        ))}
      {/* for each strategy add a radio button */}
      {strategies.map((strategy) => (
        <div key={strategy}>
          <input
            type="radio"
            name="strategy"
            value={strategy}
            onChange={handleStrategyChange}
          />
          <label>{strategy}</label>
        </div>
      ))}
      {/* if a model and the strategy are selected render the FITE component */}
      {model && strategy && (
        <FITE apiUrl={`${apiUrl}/generate/${model}?strategy=${strategy}`} />
      )}
    </div>
  );
}

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById("root")
);

// Hot Module Replacement (HMR) - Remove this snippet to remove HMR.
// Learn more: https://snowpack.dev/concepts/hot-module-replacement
if (import.meta.hot) {
  import.meta.hot.accept();
}
export default FITE;
