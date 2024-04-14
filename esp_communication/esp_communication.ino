#include "esp_camera.h"
#include <WiFi.h>

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Replace with your network credentials
const char* ssid = "#11";
const char* password = "HuizeSkipBidi!11";

const char* streamUrl = "/stream";

void setup() {
  Serial.begin(115200);

  // Initialize camera
  camera_config_t config;
  config.pin_pwdn = PWDN_GPIO_NUM;      // GPIO32
  config.pin_reset = RESET_GPIO_NUM;    // unused
  config.pin_xclk = XCLK_GPIO_NUM;      // GPIO0
  config.pin_sscb_sda = SIOD_GPIO_NUM;  // GPIO26
  config.pin_sscb_scl = SIOC_GPIO_NUM;  // GPIO27
  config.pin_d7 = Y9_GPIO_NUM;          // GPIO35
  config.pin_d6 = Y8_GPIO_NUM;          // GPIO34
  config.pin_d5 = Y7_GPIO_NUM;          // GPIO39
  config.pin_d4 = Y6_GPIO_NUM;          // GPIO36
  config.pin_d3 = Y5_GPIO_NUM;          // GPIO21
  config.pin_d2 = Y4_GPIO_NUM;          // GPIO19
  config.pin_d1 = Y3_GPIO_NUM;          // GPIO18
  config.pin_d0 = Y2_GPIO_NUM;          // GPIO5
  config.pin_vsync = VSYNC_GPIO_NUM;    // GPIO25
  config.pin_href = HREF_GPIO_NUM;      // GPIO23
  config.pin_pclk = PCLK_GPIO_NUM;      // GPIO22

  // Init Camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.print("Connected to WiFi:   ");
  Serial.println(WiFi.localIP());
}

void loop() {
  WiFiClient client;

  if (!client.connect("192.168.1.180", 80)) {
    Serial.println("Connection failed.");
    delay(1000);
    return;
  }

  // Send HTTP request to start streaming
  client.print(String("GET ") + streamUrl + " HTTP/1.1\r\n" + "Host: your_python_server_ip\r\n" + "Connection: close\r\n\r\n");
  delay(100);

  // Close client
  client.stop();
  delay(1000);
}
