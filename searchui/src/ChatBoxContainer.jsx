import React, { useState } from "react";
import { TextareaAutosize as BaseTextareaAutosize } from "@mui/base/TextareaAutosize";
import { styled } from "@mui/system";
// @ts-ignore
import enterIcon from "../assets/right-chevron.png";
import { sampleResponse } from "./response";
import { ChatBotRelay } from "./ChatBotRelay";
import { FaCircleChevronRight } from "react-icons/fa6";
import { PiSpinnerGap } from "react-icons/pi";
import axios from "axios";

const blue = {
  100: "#DAECFF",
  200: "#b6daff",
  400: "#3399FF",
  500: "#007FFF",
  600: "#0072E5",
  900: "#003A75",
};

const grey = {
  50: "#F3F6F9",
  100: "#E5EAF2",
  200: "#DAE2ED",
  300: "#C7D0DD",
  400: "#B0B8C4",
  500: "#9DA8B7",
  600: "#6B7A90",
  700: "#434D5B",
  800: "#303740",
  900: "#1C2025",
};

const Textarea = styled(BaseTextareaAutosize)(
  ({ theme }) => `
  box-sizing: border-box;
  width: 500px;
  font-family: "Roboto", sans-serif;
  font-weight: 400;
  font-style: normal;
  font-size: 0.875rem;
  line-height: 1.5;
  padding: 12px;
  border-radius: 12px 12px 0 12px;
  color: ${theme.palette.mode === "dark" ? grey[300] : grey[900]};
  background: ${theme.palette.mode === "dark" ? grey[900] : "#fff"};
  border: 1px solid ${theme.palette.mode === "dark" ? grey[700] : grey[200]};
  box-shadow: 0px 2px 2px ${
    theme.palette.mode === "dark" ? grey[900] : grey[50]
  };

  &:hover {
    border-color: ${blue[400]};
  }

  &:focus {
    outline: 0;
    border-color: ${blue[400]};
    box-shadow: 0 0 0 3px ${
      theme.palette.mode === "dark" ? blue[600] : blue[200]
    };
  }

  // firefox
  &:focus-visible {
    outline: 0;
  }
`
);

export const ChatBoxContainer = (props) => {
  const { homePage } = props;
  const [isFocused, setFocused] = useState(false);
  const [currentPrompt, setCurrentPromt] = useState("");
  const [allPrompts, setAllPrompts] = useState([]);
  const [showResponse, setShowResponse] = useState(false);
  const [allResponses, setAllResponses] = useState([]);
  const [isFirstRender, setFirstRender] = useState(true);
  const [loading, setLoading] = useState(false);

  const handleValueChange = (event) => {
    setFocused(true);
    setCurrentPromt(event.target.value);
  };
  const callSearchAPI = async () => {
    const allQns = [...allPrompts];
    allQns.push(currentPrompt);
    //getGPTResponse/${currentPrompt}
    setLoading(true);
    homePage(false);
    axios
      .get(`http://localhost:8005/prompt/query`, {
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        params: {
          prompt: currentPrompt,
        },
        timeout: 25000,
      })
      .then((response) => {
        setAllPrompts(allQns);
        setShowResponse(true);
        const allResp = [...allResponses];
        allResp.push(response.data?.message);
        setAllResponses(allResp);
        // homePage(false);
        setFirstRender(false);
        setLoading(false);
      });
  };
  return (
    <>
      {!isFirstRender && !loading && (
        <div className="chat-box-wrapper">
          {showResponse &&
            allPrompts.map((userPromt, index) => {
              return (
                <ChatBotRelay
                  prompt={userPromt}
                  response={allResponses[index]}
                />
              );
            })}
        </div>
      )}
      {!loading && (
        <div className="prompt-container">
          <Textarea
            aria-label="empty textarea"
            placeholder="Ask me anything ..."
            minRows={3}
            maxRows={10}
            onChange={handleValueChange}
          />
          {isFocused && currentPrompt?.length > 0 && (
            <div onClick={callSearchAPI}>
              {/* <img src={enterIcon} width="20" height="20"></img> */}
              <FaCircleChevronRight color={"#3399FF"} size={25} />
            </div>
          )}
        </div>
      )}
      {loading && (
        <div className="loading-icon-wrapper">
          <PiSpinnerGap size={60} color={"#9b9a9e"} />
        </div>
      )}
    </>
  );
};
