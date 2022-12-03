import React from "react";
import debounce from "lodash.debounce";
import useFite from "./hooks";
import "./style.css";

const ACTIONS = {
  TAB: "Tab",
  ENTER: "Enter",
  contain: function (key: string) {
    return Object.values(this)
      .filter((v) => typeof v === "string")
      .includes(key);
  } as (key: string) => boolean,
} as const;

enum ClassNames {
  inputLayer = "fite-input-layer",
  annotationLayer = "fite-annotation-layer",
  textAreaStack = "fite-text-area-stack",
}

export function InputLayer(): JSX.Element {
  const { dispatchState, generateText, ...state } = useFite();

  const handelTextAreaKeyDown = React.useCallback<(e: React.KeyboardEvent<HTMLTextAreaElement>) => void>(
    // event handler for the text area manages the intellisense options and the text area value
    (e) => {
      // unpack state variables
      let { textCompletion, textAreaValue } = state;
      // continue if key not in ActionKeys or textCompletion is undefined
      if (!ACTIONS.contain(e.key) || !textCompletion || !textAreaValue) return void 0;
      // prevent default action for ACTION keys
      e.preventDefault();
      // unpack the HTMLTextAreaElementEvent
      const { key, ctrlKey, currentTarget: { value: textPrompt } } = e; // prettier-ignore
      // slice the completion by the length of the prompt to get just the generated text
      textCompletion = textCompletion.slice(textPrompt.length);
      // logic...
      // [ TAB ] -> update the targetValue with the first word of the textCompletion
      if (key === ACTIONS.TAB) {
        // the index of the first space in the textCompletion that is not the first character
        const indexOfEndOfFirstWord = textCompletion.indexOf(" ", textCompletion.startsWith(" ") ? 1 : 0);
        // the first whole word in the textCompletion
        textAreaValue += textCompletion.substring(0, indexOfEndOfFirstWord);
      } // [ CTRL + ENTER ] -> full textCompletion
      else if (ctrlKey && key === ACTIONS.ENTER) {
        textAreaValue += textCompletion;
      } // [ ENTER ] -> first full line of textCompletion
      else if (key === ACTIONS.ENTER) {
        textAreaValue += textCompletion.substring(0, textCompletion.length).split("\n")[0] + "\n";
      } //
      else throw new Error(`key ${key} is not a valid ${ACTIONS}; refer to ACTIONS`);
      dispatchState({ textAreaValue });
    },
    [state.textCompletion, state.textAreaValue]
  );

  /** debounce the generateText function to prevent spamming the server*/
  const debounceCallback = React.useCallback(
    debounce(
      (text: string) => {
        generateText(text, state.model, state.strategy).then((completion) =>
          dispatchState({ textCompletion: completion.join("\n") })
        );
      },
      500,
      { leading: false, trailing: true }
    ),
    [state.model, state.strategy]
  );

  const handleOnTextAreaChange = React.useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const { value: textAreaValue } = e.target;
      if (!state.textAreaValue || !state.textAreaValue.startsWith(textAreaValue)) {
        debounceCallback(textAreaValue);
      }

      dispatchState({ textAreaValue });
    },
    [debounceCallback, state.textAreaValue]
  );

  return (
    <textarea
      className={ClassNames.inputLayer}
      onKeyDown={handelTextAreaKeyDown}
      onChange={handleOnTextAreaChange}
      value={state.textAreaValue}
    />
  );
}

export function AnnotationLayer(): JSX.Element {
  const { textCompletion } = useFite();
  return (
    <div style={{ position: "absolute" }}>
      <pre className={ClassNames.annotationLayer}>{textCompletion}</pre>
    </div>
  );
}

export function TextAreaStack({ children, ...props }): JSX.Element {
  return (
    <div className={ClassNames.textAreaStack} {...props}>
      {children}
    </div>
  );
}

export default function () {
  /* a text area with intellisense autocomplete
    that is used for writing a terminal aerodrome forecast
    the intellisense is based the TAF_STRING above
    the user can cycle through the intellisense options with the up and down arrow keys
    the user can select an option by pressing tab or enter */

  return (
    <TextAreaStack>
      <AnnotationLayer />
      <InputLayer />
    </TextAreaStack>
  );
}
