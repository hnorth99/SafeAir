//
//  LocationMetrics.swift
//  SafeAirDevice
//
//  Created by Rafka Daou on 2/24/22.
//

import Foundation
import UIKit
import FirebaseAuth
import FirebaseDatabase

class LocationMetrics: UIViewController {
    let appDelegate = UIApplication.shared.delegate as! AppDelegate
    
    @IBOutlet weak var locationLabel: UILabel!

    @IBOutlet weak var capacityLabel: UILabel!
    @IBOutlet weak var tempLabel: UILabel!
    @IBOutlet weak var humidityLabel: UILabel!
    @IBOutlet weak var co2Label: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        locationLabel.text = appDelegate.location
        readData()
        
        // Do any additional setup after loading the view.
    }
    
    func readData() {
        appDelegate.ref.child(appDelegate.location).observe(DataEventType.value, with: {
                (snapshot) in
            let dict = snapshot.value as? NSDictionary
            let capacity = dict?["capacity"] as? Double ?? 0.0
            let temperature = dict?["temperature"] as? Double ?? 0.0
            let humidity = dict?["humidity"] as? Double ?? 0.0
            let co2 = dict?["co2"] as? Double ?? 0.0
            
            self.capacityLabel.text = String(capacity)
            self.tempLabel.text = String(temperature)
            self.humidityLabel.text = String(humidity)
            self.co2Label.text = String(co2)
            
            if (capacity > 1) {
                capacityLabel.textColor = UIColor.red
            }
            if (temperature < 11) {
                tempLabel.textColor = UIColor.red
            }
            if (humidity < 40 || humidity > 60) {
                humidityLabel.textColor = UIColor.red
            }
            if (co2 > 1500) {
                co2Label.textColor = UIColor.red
            }
        }
        )
        
    }
    
}
