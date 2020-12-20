//
//  ViewController.swift
//  LunarOlympics
//
//  Created by Zheyuan Xu on 12/19/20.
//

import UIKit
import RealityKit
import ARKit
import CoreBluetooth
import CoreML
import FirebaseDatabase

class ViewController: UIViewController, CBCentralManagerDelegate, ARSessionDelegate {
    var sports: String!
    var sceneInd: Int!
    
    // initialization for firebase objects
    var rootRef: DatabaseReference!
    var conditionRef: DatabaseReference!
    
    // initialization for IMU sensor and BLE
    let IMUUUID = CBUUID(string: "917649A0-D98E-11E5-9EEC-0002A5D5C51B")
    let IMUx = CBUUID(string: "917649A0-D98E-11E5-9EEC-0002A5D5C51B")
    let IMUy = CBUUID(string: "917649A0-D98E-11E5-9EEC-0002A5D5C51C")
    let IMUz = CBUUID(string: "917649A0-D98E-11E5-9EEC-0002A5D5C51D")
    
    var accel_x: Float!
    var accel_y: Float!
    var accel_z: Float!
    
    var semaphoreX: Bool!
    var semaphoreY: Bool!
    
    var IMUPeripheral: CBPeripheral!
    var centralManager: CBCentralManager!
    
    @IBOutlet var arView: ARView!
    
    var SceneAnchor: LunarScene.Running!
    var spaceAnchor: LunarScene.Flying!
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        arView.session.delegate = self
        setupARView()
        arView.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(handleTap(recognizer:))))
        
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        sceneInd = 1
        rootRef = Database.database().reference()
        conditionRef = rootRef.child("SceneSwitch")
        
        sports = "running"
        
        semaphoreX = false
        semaphoreY = false
        
        accel_x = 0
        accel_y = 0
        accel_z = 0
        
        centralManager = CBCentralManager(delegate: self, queue: nil)
        
        
        // Load the "Box" scene from the "Experience" Reality File
        SceneAnchor = try! LunarScene.loadRunning()
        
        spaceAnchor = try! LunarScene.loadFlying()
        // Add the box anchor to the scene
        arView.scene.anchors.append(SceneAnchor)
    }
    
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .unknown:
          print("central.state is .unknown")
        case .resetting:
          print("central.state is .resetting")
        case .unsupported:
          print("central.state is .unsupported")
        case .unauthorized:
          print("central.state is .unauthorized")
        case .poweredOff:
          print("central.state is .poweredOff")
        case .poweredOn:
            print("central.state is .poweredOn")
            centralManager.scanForPeripherals(withServices: [IMUUUID])
        @unknown default:
            fatalError()
        }
    }
    
    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        print("Connected!")
        IMUPeripheral.discoverServices([IMUUUID])
        
    }
    
    func setupARView() {
        arView.automaticallyConfigureSession = false
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        configuration.environmentTexturing = .automatic
        arView.session.run(configuration)
    }
    
    // function for handling tapping actions
    @objc
    func handleTap(recognizer: UITapGestureRecognizer) {
        getQuery()
        
    }
    
    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String : Any], rssi RSSI: NSNumber) {
        print(peripheral)
        IMUPeripheral = peripheral
        IMUPeripheral.delegate = self
        centralManager.stopScan()
        centralManager.connect(IMUPeripheral)
    }
    
    func bytesToFloat(bytes b: [UInt8]) -> Float {
        let littleEndianValue = b.withUnsafeBufferPointer {
            $0.baseAddress!.withMemoryRebound(to: UInt32.self, capacity: 1) { $0.pointee }
        }
        let bitPattern = UInt32(littleEndian: littleEndianValue)
        return Float(bitPattern: bitPattern)
    }
    
    func getQuery() {
        self.conditionRef.child("switchReq").observeSingleEvent(of: .value, with: { (snapshot) in
            // Get user value
            let value = snapshot.value as? Int
            if value == 1 {
                if self.sceneInd == 1{
                    self.SceneAnchor.notifications.switchScene.post()
                    self.sceneInd = 2
                } else {
                    self.spaceAnchor.notifications.switchScene.post()
                    self.sceneInd = 1
                }
                
            }
            }) { (error) in
              print(error.localizedDescription)
        }
    }
}


extension ViewController: CBPeripheralDelegate {
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let services = peripheral.services else { return }
        for service in services {
            peripheral.discoverCharacteristics(nil, for: service)
        }
        
    }
    
    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
      guard let characteristics = service.characteristics else { return }
      for characteristic in characteristics {
        print(characteristic)
        peripheral.readValue(for: characteristic)
        peripheral.setNotifyValue(true, for: characteristic)
      }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic,
                    error: Error?) {
      switch characteristic.uuid {
      case IMUx:
          let rawX = [UInt8](characteristic.value!)
          accel_x = bytesToFloat(bytes: rawX)
          if (accel_x < -0.7) && (!semaphoreX) {
            semaphoreX = true
            // SceneAnchor.notifications.runLeft.post()
            DispatchQueue.main.async {
                _ = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: false) {_ in
                    // release semaphore for Y axis after 0.5 seconds
                    self.semaphoreX = false
                }
            }
          }
          if (accel_x > 0.7) && (!semaphoreX) {
            semaphoreX = true
            // SceneAnchor.notifications.runRight.post()
            DispatchQueue.main.async {
                _ = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: false) {_ in
                    // release semaphore for Y axis after 0.5 seconds
                    self.semaphoreX = false
                }
            }
          }
          //print(accel_x ?? "no value for x-axis acceleration")
      case IMUy:
          let rawY = [UInt8](characteristic.value!)
          accel_y = bytesToFloat(bytes: rawY)
          print(accel_y ?? "no value for y-axis acceleration")
          //semaphoreY = false
          if (accel_y < -0.7) && (!semaphoreY) {
            // block further call
            semaphoreY = true
            print("function called")
            if sceneInd == 1 {
                SceneAnchor.notifications.runTrigger.post()
            } else {
                spaceAnchor.notifications.flyRight.post()
            }
            
            DispatchQueue.main.async {
                _ = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: false) {_ in
                    // release semaphore for Y axis after 0.5 seconds
                    self.semaphoreY = false
                }
            }
          }
          if (accel_y > 0.7) && (!semaphoreY){
            // block further call
            semaphoreY = true
            if sceneInd == 1 {
                SceneAnchor.notifications.jumpTrigger.post()
            } else {
                spaceAnchor.notifications.flyLeft.post()
            }
            
            DispatchQueue.main.async {
                _ = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: false) {_ in
                    // release semaphore for Y axis after 0.5 seconds
                    self.semaphoreY = false
                }
            }
          }
      case IMUz:
          let rawZ = [UInt8](characteristic.value!)
          accel_z = bytesToFloat(bytes: rawZ)
          //print(accel_z ?? "no value for y-axis acceleration")
      default:
          print("Unhandled Characteristic UUID: \(characteristic.uuid)")
      }
    }
}
