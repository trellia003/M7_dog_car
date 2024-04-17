#include <WiFiS3.h> 
#include <SoftwareSerial.h> 
SoftwareSerial mySerial(10, 11);  // RX, TX

// char ssid[] = "#11";               // your network SSID (name)
// char pass[] = "HuizeSkipBidi!11";  // your network password (use for WPA, or use as key for WEP)
 
char ssid[] = "elia";         // your network SSID (name)
char pass[] = "fantacalcio";  // your network password (use for WPA, or use as key for WEP)

int status = WL_IDLE_STATUS;

WiFiServer server(23);

boolean alreadyConnected = false;  // whether or not the client was connected previously

void setup() {
  Serial.begin(9600);
  mySerial.begin(9600);
  serverSetup();
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

    if (client.available() > 0) {
      String command = client.readStringUntil('\n');
      Serial.println("Received command:          " + command);
      serialToControlArduino(command);
    }
  }
}

void serialToControlArduino(String command) {
  mySerial.println(command);
}

void serverSetup() {

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
