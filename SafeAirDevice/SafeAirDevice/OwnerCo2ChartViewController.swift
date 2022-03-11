//  OwnerCo2ChartViewController.swift
//  SafeAirDevice
//  Created by Rafka Daou on 3/4/22.

// The following are packages imported for this program.
import UIKit
import UIKit
import Charts
import TinyConstraints
import Foundation
import FirebaseAuth
import FirebaseDatabase

// The OwnerCo2ChartViewController is designed to display the historic metrics of the carbon dioxide measurements. These measurements are fetched from the firebase real-time database and then displayed as a line chart. In addition
// the owner has the able to see customer complaints as well as edit the constraints that serve as predefined setpoints to regulating external devices.
class OwnerCo2ChartViewController: UIViewController, ChartViewDelegate {
    
    // The following code initializes UILabels and TextFields for this
    // program.
    @IBOutlet weak var editCO2Max: UITextField!
    @IBOutlet weak var currentC02Constr: UILabel!
    @IBOutlet weak var CO2Comp: UILabel!
    
    // The following code initializes a variable appDelegate.
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
        //var entries = [Double]()
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
            //var time = entries?["time"] as? [NSString] ?? [""]
            for x in (co2.count - 70)..<co2.count {
                test.append(ChartDataEntry(x: Double(x-70), y: Double(co2[x])))
            }
            
            // The following displays the current CO2 complaints.
            let complaint = entries?["co2complaints"] as? Double ?? 0.0005
            self.CO2Comp.text = String(complaint)
            
            // The following displays the current CO2 constraints.
            let CO2con = entries?["co2_max"] as? String ?? ""
            self.currentC02Constr.text = String(CO2con) + " ppm"

            
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
    // The following function apply saves the newly updated CO2 max constraint to the database.
    @IBAction func apply_pressed(_ sender: Any) {
        appDelegate.ref.child("HunterApt").child("co2_max").setValue(editCO2Max.text);
    }
}
