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
  right_back();
  right_front();
  left_back();
  left_front();

  //THE BACK ARE INVERTED!!!!

}

void loop() {
  // put your main code here, to run repeatedly:
}




void right_back(){
    // put your setup code here, to run once:
  pinMode(ENA_RB, OUTPUT);
  pinMode(IN1_RB, OUTPUT);
  pinMode(IN2_RB, OUTPUT);



  //forward
  digitalWrite(IN1_RB, HIGH);
  digitalWrite(IN2_RB, LOW);
  digitalWrite(ENA_RB, 100);
  delay(3000);

  digitalWrite(IN1_RB, LOW);
  digitalWrite(IN2_RB, HIGH);
  digitalWrite(ENA_RB, 100);
  delay(3000);



  digitalWrite(IN1_RB, LOW);
  digitalWrite(IN2_RB, LOW);
  digitalWrite(ENA_RB, 0);
}




void right_front(){
    // put your setup code here, to run once:
  pinMode(ENA_RF, OUTPUT);
  pinMode(IN3_RF, OUTPUT);
  pinMode(IN4_RF, OUTPUT);



  //forward
  digitalWrite(IN3_RF, HIGH);
  digitalWrite(IN4_RF, LOW);
  digitalWrite(ENA_RF, 100);
  delay(3000);

  digitalWrite(IN3_RF, LOW);
  digitalWrite(IN4_RF, HIGH);
  digitalWrite(ENA_RF, 100);
  delay(3000);



  digitalWrite(IN3_RF, LOW);
  digitalWrite(IN4_RF, LOW);
  digitalWrite(ENA_RF, 0);
}



void left_back(){
    // put your setup code here, to run once:
  pinMode(ENA_LB, OUTPUT);
  pinMode(IN3_LB, OUTPUT);
  pinMode(IN4_LB, OUTPUT);



  //forward
  digitalWrite(IN3_LB, HIGH);
  digitalWrite(IN4_LB, LOW);
  digitalWrite(ENA_LB, 100);
  delay(3000);

  digitalWrite(IN3_LB, LOW);
  digitalWrite(IN4_LB, HIGH);
  digitalWrite(ENA_LB, 100);
  delay(3000);



  digitalWrite(IN3_LB, LOW);
  digitalWrite(IN4_LB, LOW);
  digitalWrite(ENA_LB, 0);
}



void left_front(){
    // put your setup code here, to run once:
  pinMode(ENA_LF, OUTPUT);
  pinMode(IN1_LF, OUTPUT);
  pinMode(IN2_LF, OUTPUT);



  //forward
  digitalWrite(IN1_LF, HIGH);
  digitalWrite(IN2_LF, LOW);
  digitalWrite(ENA_LF, 100);
  delay(3000);

  digitalWrite(IN1_LF, LOW);
  digitalWrite(IN2_LF, HIGH);
  digitalWrite(ENA_LF, 100);
  delay(3000);



  digitalWrite(IN1_LF, LOW);
  digitalWrite(IN2_LF, LOW);
  digitalWrite(ENA_LF, 0);
}