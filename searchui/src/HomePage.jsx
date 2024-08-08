import React, { useState } from "react";
import { LandingPage } from "./LandingPage";
import { ChatBoxContainer } from "./ChatBoxContainer";
import { Header } from "./Header";
import sampleUserImg from "../assets/sample-user-img.jpg";

const mockUser = {
  name: "Catherine",
  img: sampleUserImg,
};

export const HomePage = () => {
  const [showLandingPage, setShowLandingPage] = useState(true);
  return (
    <div className="compass-home-container">
      <Header user={mockUser} />
      {showLandingPage && <LandingPage />}
      <ChatBoxContainer homePage={setShowLandingPage} />
    </div>
  );
};
