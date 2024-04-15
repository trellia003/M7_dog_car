#include <SoftwareSerial.h>

// pin definitions for the motorshield
//B is left
//A is right
#define BRAKE_A 9
#define DIR_A 12
#define PWM_A 3
#define BRAKE_B 8
#define DIR_B 13
#define PWM_B 11

unsigned long int loopTime = 0;

//if you want null by default, move this to readCommand, then make read command a return string and pass the string to motor decision
String command = "";
String speed = "";

SoftwareSerial mySerial(5, -1);  // RX, TX (only receiving) on pin 5

void setup() {
  Serial.begin(9600);    // Initialize serial communication
  mySerial.begin(9600);  // Initialize your serial communication port

  pinMode(BRAKE_A, OUTPUT);  // brake
  pinMode(DIR_A, OUTPUT);    // dir
  pinMode(PWM_A, OUTPUT);    // PWM
  pinMode(BRAKE_B, OUTPUT);  // brake
  pinMode(DIR_B, OUTPUT);    // dir
  pinMode(PWM_B, OUTPUT);    // PWM
}

void loop() {
  if (millis() > loopTime + 49) {  // 20 Hz output
    loopTime = millis();           // reset looptimer
    readCommand();
    motorDecision();
  }
}

void readCommand() {
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
      String buffer = receivedString.substring(startPos, endPos);
      int delimiterIndex = buffer.indexOf(':');
      command = buffer.substring(0, delimiterIndex);  // Extract the first part
      speed = buffer.substring(delimiterIndex + 1);
      // Serial.print("Extracted string: ");
      // Serial.println(command); // Print the extracted string
      // Here you can further process the extracted string
    }
  }
}

void motorDecision() {
  if (command == "stop") {
    stop();
    // Serial.println("stop");
  } else if (command == "pvotleft") {
    pvotleft();
    // Serial.println("left"+ speed);
  } else if (command == "pvotright") {
    pvotright();
    // Serial.println("right" + speed);
  } else if (command == "onspotleft") {
    onspotleft();
    // Serial.println("left"+ speed);
  } else if (command == "onspotright") {
    onspotright();
    // Serial.println("right" + speed);
  } else if (command == "forward") {
    forward();
    // Serial.println("forward" + speed);
  } else {
    Serial.println(command + speed);
  }
}
void stop() {
  setMotor_A(0);
  setMotor_B(0);
}

void forward() {
  setMotor_A(speed.toInt());
  setMotor_B(speed.toInt());
}

void pvotleft() {
  setMotor_A(speed.toInt());
  setMotor_B(0);
}
void pvotright() {
  setMotor_A(0);
  setMotor_B(speed.toInt());
}

void onspotleft() {
  setMotor_A(speed.toInt());
  setMotor_B(-speed.toInt());
}

void onspotright() {
  setMotor_A(-speed.toInt());
  setMotor_B(speed.toInt());
}



void setMotor_A(int value) {
  if (value > 0) {
    digitalWrite(DIR_A, LOW);
    digitalWrite(BRAKE_A, LOW);
    analogWrite(PWM_A, min(value, 255));
  } else if (value < 0) {
    digitalWrite(DIR_A, HIGH);
    digitalWrite(BRAKE_A, LOW);
    analogWrite(PWM_A, min(abs(value), 255));
  } else {
    digitalWrite(DIR_A, LOW);
    digitalWrite(BRAKE_A, HIGH);
    analogWrite(PWM_A, 0);
  }
}

// Set motor, use brake when 0 is sent
void setMotor_B(int value) {
  if (value > 0) {
    digitalWrite(DIR_B, HIGH);
    digitalWrite(BRAKE_B, LOW);
    analogWrite(PWM_B, min(value, 255));
  } else if (value < 0) {
    digitalWrite(DIR_B, LOW);
    digitalWrite(BRAKE_B, LOW);
    analogWrite(PWM_B, min(abs(value), 255));
  } else {
    digitalWrite(DIR_B, LOW);
    digitalWrite(BRAKE_B, HIGH);
    analogWrite(PWM_B, 0);
  }
}
