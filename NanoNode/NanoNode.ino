#include <Arduino.h>

# define FAN_PIN     PD2 
# define DEHUMID_PIN PD3
# define HUMID_PIN   PD4
# define HEATER_PIN  PD5

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    delay(100);
  }  

  pinMode(HEATER_PIN, OUTPUT); // BLUE
  pinMode(HUMID_PIN, OUTPUT);  // GREEN
  pinMode(DEHUMID_PIN, OUTPUT);// YELLOW
  pinMode(FAN_PIN, OUTPUT);    // RED
} 

void loop() {
  // Read response from HC-06
  if (Serial.available() > 0) {
    int response = Serial.parseInt();
    if (response % 10 == 1) {
      Serial.println("fan on");
      digitalWrite(FAN_PIN, HIGH);
    } else {
      digitalWrite(FAN_PIN, LOW);
    }
    if ((response / 10) % 10 == 1) {
      digitalWrite(DEHUMID_PIN, HIGH);
      Serial.println("dehum on");
    } else {
      digitalWrite(DEHUMID_PIN, LOW);
    }
    if ((response / 100) % 10 == 1) {
      digitalWrite(HUMID_PIN, HIGH);
      Serial.println("hum on");
    } else {
      digitalWrite(HUMID_PIN, LOW);
    }
    if ((response / 1000) % 10 == 1) {
      digitalWrite(HEATER_PIN, HIGH);
      Serial.println("heater on");
    } else {
      digitalWrite(HEATER_PIN, LOW);
    }
  }
  delay(1000);
}
