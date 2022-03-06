//
//  OwnerCo2ChartViewController.swift
//  SafeAirDevice
//
//  Created by Rafka Daou on 3/4/22.
//

import UIKit
import UIKit
import Charts
import TinyConstraints
import Foundation
import FirebaseAuth
import FirebaseDatabase

class OwnerCo2ChartViewController: UIViewController, ChartViewDelegate {
    
    @IBOutlet weak var editCO2Max: UITextField!

    @IBOutlet weak var currentC02Constr: UILabel!
    @IBOutlet weak var CO2Comp: UILabel!
    let appDelegate = UIApplication.shared.delegate as! AppDelegate
    var lineChart = LineChartView()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        lineChart.delegate = self
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        lineChart.frame = CGRect(x:0, y:0, width: self.view.frame.size.width,
                                 height: self.view.frame.size.width)
        lineChart.center = view.center
        view.addSubview(lineChart)
        // The varaible test is created of type 'ChartDataEntry' that will hold the
        // values that will then be plotted.
        var test = [ChartDataEntry]()
        var entries = [Double]()
        // The following appDelegate function indexes into the node 'HunterApt to get
        // access to all the associated children.
        appDelegate.ref.child("HunterApt").getData(completion: { error, snapshot in
            guard error == nil else {
                print(error!.localizedDescription)
                return;
            }
            // 'co2' and 'Time' are then the selected child node of Hunter Apt.
            // These values will serve as the x and y points for the graph configured.
            let entries = snapshot.value as? NSDictionary
            let co2 = entries?["co2"] as? [Double] ?? [0.0005]
            var time = entries?["time"] as? [NSString] ?? [""]
            let complaint = entries?["co2complaints"] as? Double ?? 0.0005
            self.CO2Comp.text = String(complaint)
            let CO2con = entries?["co2_max"] as? String ?? ""
            self.currentC02Constr.text = String(CO2con)
            for x in 0..<70 {
                test.append(ChartDataEntry(x: Double(x-70), y: Double(co2[x])))
            }
            // The following code creates modifications to the graph being displayed.
            // For example eliminating the xAxis labels and only having the Y axis labels
            // appear on one side of the graph instead of both.
            self.lineChart.xAxis.drawLabelsEnabled = false
            self.lineChart.rightAxis.drawLabelsEnabled = false
            self.lineChart.legend.enabled = false
            
            // The following line plots the data in test onto the graph.
            let set = LineChartDataSet(entries: test)
            
            // The following color modifications are made to the graph.
            set.colors = ChartColorTemplates.material()
            let data = LineChartData(dataSet: set)
            self.lineChart.data = data
        }
        )
    }
    @IBAction func apply_pressed(_ sender: Any) {
        appDelegate.ref.child("HunterApt").child("co2_max").setValue(editCO2Max.text);
    }
}
