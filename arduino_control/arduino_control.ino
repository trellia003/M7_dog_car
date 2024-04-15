#include <SoftwareSerial.h>


SoftwareSerial mySerial(5, -1);  // RX, TX (only receiving) on pin 5

void setup() {
  Serial.begin(9600);    // Initialize serial communication
  mySerial.begin(9600);  // Initialize your serial communication port
} 

void loop() {
  Serial.println(readCommand());
  delay (200);
}

String readCommand() {
  String command = "";
  if (mySerial.available() > 0) {                            // Check if data is available to read
    String receivedString = mySerial.readStringUntil('\n');  // Read the incoming data until newline character '\n'
    // Serial.print("Received string: ");
    // Serial.println(receivedString); // Print the received string

    // Find the position of "com:" and ":end" in the received string
    int startPos = receivedString.indexOf("Com:") + 4;  // Add 4 to move past "Com:"
    int endPos = receivedString.indexOf(":end");

    // If both "Com:" and ":end" are found
    if (startPos != -1 && endPos != -1) {
      // Extract the substring between "Com:" and ":end"
      command = receivedString.substring(startPos, endPos);
      // Serial.print("Extracted string: ");
      // Serial.println(command); // Print the extracted string
      // Here you can further process the extracted string
    }
  }
  return command;
}