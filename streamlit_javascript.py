import React from 'react';
import { Circle } from 'lucide-react';

const LED = ({ isOn, index }) => (
  <div className="relative group">
    <Circle 
      className={`w-16 h-16 transition-all duration-500 ${
        isOn 
          ? 'text-yellow-400 drop-shadow-[0_0_8px_rgba(250,204,21,0.8)]' 
          : 'text-gray-300'
      }`}
      strokeWidth={2}
      fill={isOn ? 'currentColor' : 'none'}
    />
    <span className="absolute -bottom-6 left-1/2 transform -translate-x-1/2 text-sm text-gray-500">
      LED {index + 1}
    </span>
  </div>
);

const LEDDisplay = ({ activeCount = 0 }) => {
  return (
    <div className="bg-slate-900 p-8 rounded-xl shadow-2xl">
      <div className="flex justify-center items-center space-x-8 mb-6">
        {[...Array(5)].map((_, i) => (
          <LED key={i} isOn={i < activeCount} index={i} />
        ))}
      </div>
      <div className="text-center text-gray-400 mt-8">
        <p className="text-lg">Active LEDs: {activeCount} / 5</p>
      </div>
    </div>
  );
};

export default LEDDisplay;
