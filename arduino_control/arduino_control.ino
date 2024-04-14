#include <SoftwareSerial.h>

const int rxPin = 11;  // RX pin of Arduino 2 connected to TX pin of Arduino 1
SoftwareSerial mySerial(rxPin, -1);  // RX, TX

void setup() {
  Serial.begin(9600);
  mySerial.begin(9600);
}

void loop() {
  if (mySerial.available()) {
    String data = mySerial.readStringUntil('\n');  // Read until newline character
    Serial.println(data);  // Print received data for debugging

    // // Parsing the received data
    // int commaIndex = data.indexOf(',');
    // if (commaIndex != -1) {
    //   String valueA = data.substring(1, commaIndex); // Extract value of A
    //   String valueB = data.substring(commaIndex + 2, data.length() - 1); // Extract value of B

    //   int intValueA = valueA.toInt();  // Convert A to integer
    //   int intValueB = valueB.toInt();  // Convert B to integer

    //   // Now you have the values of A and B in integer variables intValueA and intValueB respectively
    //   Serial.print("Value of A: ");
    //   Serial.println(intValueA);
    //   Serial.print("Value of B: ");
    //   Serial.println(intValueB);
    // }
  }
}
