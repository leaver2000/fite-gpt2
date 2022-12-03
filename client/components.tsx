import React from "react";
import debounce from "lodash.debounce";
import useFite from "./hooks";
import "./style.css";

enum ActionKey {
  TAB = "Tab",
  ENTER = "Enter",
  // UP = "ArrowUp",
  // DOWN = "ArrowDown",
  // BACKSPACE = "Backspace",
}
const ActionKeys = Object.values(ActionKey) as string[];

function FITE() {
  /* a text area with intellisense autocomplete
    that is used for writing a terminal aerodrome forecast
    the intellisense is based the TAF_STRING above
    the user can cycle through the intellisense options with the up and down arrow keys
    the user can select an option by pressing tab or enter */
  const { dispatchState, generateText, ...state } = useFite();
  const { model, strategy, textPrompt, textCompletion } = state;

  const handleKeyDown = React.useCallback<(e: React.KeyboardEvent<HTMLTextAreaElement>) => void>(
    // event handler for the text area manages the intellisense options and the text area value
    (e) => {
      let { textAreaValue } = state;
      // cannot perform any action if there is no text completion or textAreaValue
      if (!textCompletion || !textAreaValue) return void 0;
      // unpack the HTMLTextAreaElementEvent
      const { key, ctrlKey, currentTarget } = e;
      // and currentTarget.value as targetValue
      let { value } = currentTarget;
      // ActionKey logic...
      if (ActionKeys.includes(key)) {
        e.preventDefault();
        // [ TAB ] -> update the targetValue with the first word of the textCompletion
        if (key === ActionKey.TAB) {
          // substring the textCompletion to the first word not in the targetValue and split it into an array
          const textList = textCompletion.substring(value.length, textCompletion.length).split(" ");
          // dispatch the new textAreaValue to the state as state.textAreaValue
          // needs formatting work ieL `TAF KBLV 141100Z 1411/1512 360[TAB]  ...`
          textAreaValue = `${value.trim()} ${textList.find((s) => !!s)}`;
          // [ CTRL + ENTER ] -> ...
        } else if (key === ActionKey.ENTER && ctrlKey) {
          textAreaValue = textCompletion;
        } else if (key === ActionKey.ENTER) {
          textAreaValue = value + textCompletion.substring(value.length, textCompletion.length).split("\n")[0] + "\n";
        } else {
          // raise an error if the key is not one of the above
          throw new Error(`key ${key} is not a valid ActionKey; refer to ActionKeys`);
        }
        dispatchState({ textAreaValue });
      }
    },
    [textPrompt, textCompletion, state.textAreaValue]
  );

  /** debounce the generateText function to prevent spamming the server*/
  const bounce = React.useCallback(
    debounce(
      (text: string) => {
        generateText(text, model, strategy).then((completion) =>
          dispatchState({ textCompletion: completion.join("\n") })
        );
      },
      1000,
      { leading: false, trailing: true }
    ),
    [model, strategy]
  );

  const handleOnTextAreaChange = React.useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const { value } = e.target;
      bounce(value);
      dispatchState({ textAreaValue: value });
    },
    [bounce]
  );

  return (
    <div className="taf-component-container">
      <div style={{ position: "absolute" }}>
        <pre className="taf-annotation-layer">{textCompletion}</pre>
      </div>
      <textarea
        className="taf-input-layer"
        onKeyDown={handleKeyDown}
        onChange={handleOnTextAreaChange}
        value={state.textAreaValue}
      />
    </div>
  );
}

export default FITE;
