
import ChatBot from "@/components/ChatBot";
import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div className="grid grid-cols-12">
      <div className="welcomeCtn col-span-4 shadow  flex flex-col items-center justify-center">
        <div className="p-2 flex flex-col items-center">
          <Image src="/logo_white.png" alt="logo" width={50} height={50} className="logo"></Image>
          <h1 className="text-3xl mb-3 mt-3 text-center"> Welcome to ChatENM </h1>
          <p className="text-justify">
            This is a prototype of the tool to assist researchers in ecological modelling.
            At it current stage the tools offers capabilities to perform ecological niche modelling
            utilizing MaxEnt. But we intend to integrate more Models in the future.
            The propose tool utilizes AI-agent to download species data on GBIF, shapefiles of countries
            and environmental variables on databases such as WordClim.
            This is done automatically by the AI-agent based on the user input.
            Outputs are downloadable tif files for the end user.
          </p>
        </div>
      </div>
      <div className="col-span-8">
        <ChatBot/>
      </div>
    </div>
  );
}
