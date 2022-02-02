#include <Arduino.h>

# define HEATER_PIN  20
# define HUMID_PIN 21
# define DEHUMID_PIN   22
# define FAN_PIN     23

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    delay(100);
  }  

  pinMode(HEATER_PIN, OUTPUT);
  pinMode(HUMID_PIN, OUTPUT);
  pinMode(DEHUMID_PIN, OUTPUT);
  pinMode(FAN_PIN, OUTPUT);
} 

void loop() {
  // Read response from HC-06
  if (Serial.available() > 0) {
    String response = Serial.readString();
    if (response.charAt(0) == '1') {
      digitalWrite(HEATER_PIN, HIGH);
    } else {
      digitalWrite(HEATER_PIN, LOW);
    }
    if (response.charAt(1) == '1') {
      digitalWrite(HUMID_PIN, HIGH);
    } else {
      digitalWrite(HUMID_PIN, LOW);
    }
    if (response.charAt(2) == '1') {
      digitalWrite(DEHUMID_PIN, HIGH);
    } else {
      digitalWrite(DEHUMID_PIN, LOW);
    }
    if (response.charAt(3) == '1') {
      digitalWrite(FAN_PIN, HIGH);
    } else {
      digitalWrite(FAN_PIN, LOW);
    }
  }
  delay(10000);
}
