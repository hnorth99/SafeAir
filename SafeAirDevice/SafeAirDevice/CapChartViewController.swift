//  CapChartViewController.swift
//  SafeAirDevice
//  Created by Rafka Daou on 2/28/22.

import UIKit
import Charts
import TinyConstraints
import Foundation
import FirebaseAuth
import FirebaseDatabase

class CapChartViewController: UIViewController, ChartViewDelegate {
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
            // 'maxcapacity' and 'Time' are then the selected child node of Hunter Apt.
            // These values will serve as the x and y points for the graph configured.
            let entries = snapshot.value as? NSDictionary
            let cap = entries?["currentcapacity"] as? Double ?? 0.0005
            var time = entries?["time"] as? [NSString] ?? [""]
            //for x in 0..<cap.count {
            //    test.append(ChartDataEntry(x: Double(x-70), y: Double(cap[x])))
            //}
            
            test.append(ChartDataEntry(x: Double(1), y: Double(cap)))
            self.lineChart.xAxis.drawLabelsEnabled = false
            self.lineChart.rightAxis.drawLabelsEnabled = false
            self.lineChart.legend.enabled = false
            let set = LineChartDataSet(entries: test)
            set.colors = ChartColorTemplates.material()
            let data = LineChartData(dataSet: set)
            self.lineChart.data = data
        }
        )
    }
    @IBAction func send_complaint(_ sender: Any) {
        appDelegate.ref.child("HunterApt").getData(completion: { error, snapshot in
            guard error == nil else {
                print(error!.localizedDescription)
                return;
            }
            // 'maxcapacity' and 'Time' are then the selected child node of Hunter Apt.
            // These values will serve as the x and y points for the graph configured.
            let entries = snapshot.value as? NSDictionary
            let complaints = entries?["capacitycomplaints"] as? Double ?? 0.0005
            self.appDelegate.ref.child("HunterApt").child("capacitycomplaints").setValue(complaints + 1.0);
        }
        )
    }
}
