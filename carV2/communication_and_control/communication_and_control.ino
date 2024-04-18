#include <WiFiS3.h>
char ssid[] = "elia";         // your network SSID (name)
char pass[] = "fantacalcio";  // your network password
WiFiServer server(23);
boolean alreadyConnected = false;  // whether or not the client was connected previously


// Define maximum command length
const int MAX_COMMAND_LENGTH = 20;
WiFiClient client;
// Buffer for storing incoming commands
char commandBuffer[MAX_COMMAND_LENGTH + 1];  // +1 for null terminator
char previous_decision = 's';


#define IN3_RF 4
#define IN4_RF 2
#define ENA_RF 3
#define IN1_LF 7
#define IN2_LF 5
#define ENA_LF 6
#define IN1_RB 12
#define IN2_RB 13
#define ENA_RB 11
#define IN3_LB 10
#define IN4_LB 8
#define ENA_LB 9

void setup() {
  Serial.begin(9600);
  serverSetup();
  setup_motors();
}

void loop() {
  WiFiClient client = server.available();

  if (client) {
    if (!alreadyConnected) {
      client.flush();
      Serial.println("We have a new client");
      client.println("Hello, client!");
      alreadyConnected = true;
    }

    // String command = client.readStringUntil('\n');
    // if (!command.isEmpty()) {
    //   // Serial.println("Received command: " + command);
    //   int colonIndex = command.indexOf(':');                    // Find the position of the colon separating the command and speed
    //   if (colonIndex != -1) {                                   // If colon is found
    //     String commandPart = command.substring(0, colonIndex);  // Extract command part
    //     String speedPart = command.substring(colonIndex + 1);   // Extract speed part
    //     int speed = speedPart.toInt();
    //     motor_head_decision(commandPart, speed);
    //   }
    // }
    int commandLength = client.readBytesUntil('\n', commandBuffer, MAX_COMMAND_LENGTH);
    if (commandLength > 0) {
      commandBuffer[commandLength] = '\0';  // Null-terminate the string
      parseCommand(commandBuffer);
    }
  }
}

void parseCommand(const char* command) {
  // Find the position of the colon separating the command and speed
  const char* colonPosition = strchr(command, ':');
  if (colonPosition != NULL) {
    // Calculate the length of the command part
    size_t commandLength = colonPosition - command;

    // Create a string to store the command part
    // String commandString = String(command).substring(0, commandLength);
    char decision = *(colonPosition - 1);
    // Extract the speed part and convert it to an integer
    int speed = atoi(colonPosition + 1);  // Convert speed part to integer

    motor_head_decision(decision, speed);
  }
}


void serverSetup() {
  int status = WL_IDLE_STATUS;

  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    while (true)
      ;
  }

  String fv = WiFi.firmwareVersion();
  if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
    Serial.println("Please upgrade the firmware");
  }

  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(ssid);
    status = WiFi.begin(ssid, pass);
    delay(5000);
  }

  server.begin();
  printWifiStatus();
}

void printWifiStatus() {
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  long rssi = WiFi.RSSI();
  Serial.print("Signal strength (RSSI): ");
  Serial.print(rssi);
  Serial.println(" dBm");
}

void motor_head_decision(char decision, int speed) {
  Serial.print("newcommand:   ");
  if (previous_decision != decision) {
    stop();
    delay(200);
  }
  previous_decision = decision;

  Serial.println(decision);
  if (decision == 's') {
    stop();
  } else if (decision == 'f') {
    forward(speed);
  } else if (decision == 'b') {
    backward(speed);
  } else if (decision == 'l') {
    left(speed);
  } else if (decision == 'r') {
    right(speed);
  }
}

void setup_motors() {
  pinMode(ENA_RF, OUTPUT);
  pinMode(IN3_RF, OUTPUT);
  pinMode(IN4_RF, OUTPUT);
  pinMode(ENA_RB, OUTPUT);
  pinMode(IN1_RB, OUTPUT);
  pinMode(IN2_RB, OUTPUT);
  pinMode(ENA_LF, OUTPUT);
  pinMode(IN1_LF, OUTPUT);
  pinMode(IN2_LF, OUTPUT);
  pinMode(ENA_LB, OUTPUT);
  pinMode(IN3_LB, OUTPUT);
  pinMode(IN4_LB, OUTPUT);
}
void forward(int speed) {
  F_RF(speed);
  F_RB(speed);
  F_LF(speed);
  F_LB(speed);
}
void backward(int speed) {
  B_RF(speed);
  B_RB(speed);
  B_LF(speed);
  B_LB(speed);
}
void left(int speed) {
  F_RB(speed);
  F_RF(speed);
  B_LF(speed);
  B_LB(speed);
}
void right(int speed) {
  B_RB(speed);
  B_RF(speed);
  F_LF(speed);
  F_LB(speed);
}
void stop() {
  S_RF();
  S_RB();
  S_LF();
  S_LB();
}

void F_RF(int speed) {  //forward right front
  digitalWrite(IN3_RF, HIGH);
  digitalWrite(IN4_RF, LOW);
  analogWrite(ENA_RF, speed);
}
void B_RF(int speed) {  //backwards right front
  digitalWrite(IN3_RF, LOW);
  digitalWrite(IN4_RF, HIGH);
  analogWrite(ENA_RF, speed);
}
void S_RF() {  //stop right front
  digitalWrite(IN3_RF, LOW);
  digitalWrite(IN4_RF, LOW);
  analogWrite(ENA_RF, 0);
}
void F_RB(int speed) {  //forward right back
  digitalWrite(IN1_RB, LOW);
  digitalWrite(IN2_RB, HIGH);
  analogWrite(ENA_RB, speed);
}
void B_RB(int speed) {  //backwards right back
  digitalWrite(IN1_RB, HIGH);
  digitalWrite(IN2_RB, LOW);
  analogWrite(ENA_RB, speed);
}
void S_RB() {  //stop right back
  digitalWrite(IN1_RB, LOW);
  digitalWrite(IN2_RB, LOW);
  analogWrite(ENA_RB, 0);
}
void F_LF(int speed) {  //forward left front
  digitalWrite(IN1_LF, HIGH);
  digitalWrite(IN2_LF, LOW);
  analogWrite(ENA_LF, speed);
}
void B_LF(int speed) {  //backward left front
  digitalWrite(IN1_LF, LOW);
  digitalWrite(IN2_LF, HIGH);
  analogWrite(ENA_LF, speed);
}
void S_LF() {  //stop left front
  digitalWrite(IN1_LF, LOW);
  digitalWrite(IN2_LF, LOW);
  analogWrite(ENA_LF, 0);
}
void F_LB(int speed) {  //forward left back
  digitalWrite(IN3_LB, LOW);
  digitalWrite(IN4_LB, HIGH);
  analogWrite(ENA_LB, speed);
}
void B_LB(int speed) {  //backward left back
  digitalWrite(IN3_LB, HIGH);
  digitalWrite(IN4_LB, LOW);
  analogWrite(ENA_LB, speed);
}
void S_LB() {  //stop left bcak
  digitalWrite(IN3_LB, LOW);
  digitalWrite(IN4_LB, LOW);
  analogWrite(ENA_LB, 0);
}