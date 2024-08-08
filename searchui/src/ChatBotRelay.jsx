import React from "react";
import { FaUser } from "react-icons/fa";
import { RiRobot2Fill } from "react-icons/ri";

export const ChatBotRelay = (props) => {
  const { prompt, response } = props;
  return (
    <div className="response-chat-container">
      <div className="user-prompt-box">
        <div className="user-icon-text">
          <span>
            <FaUser />
          </span>
          <span>You</span>
        </div>
        <div>{prompt}</div>
      </div>
      <div className="bot-response-box">
        <div className="bot-resp-head">
          <span>
            <RiRobot2Fill />
          </span>
          <span>Compass</span>
        </div>
        <div>{response}</div>
      </div>
    </div>
  );
};
