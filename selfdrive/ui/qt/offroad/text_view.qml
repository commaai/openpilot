import QtQuick 2.0

Item {
  id: root
  signal qmlSignal()

  Rectangle {
    color: "black"
    anchors.fill: parent
  }

  Flickable {
    id: flickArea
    objectName: "flickArea"
    anchors.fill: parent
    contentHeight: helpText.height
    contentWidth: helpText.width
    flickableDirection: Flickable.VerticalFlick
    flickDeceleration: 7500.0
    maximumFlickVelocity: 10000.0
    pixelAligned: true

    onAtYEndChanged: root.qmlSignal()

    Text {
      // HTML like markup can also be used
      id: helpText
      width: flickArea.width
      font.pointSize: font_size
      textFormat: Text.RichText
      color: "white"
      wrapMode: Text.Wrap

      text: tc_html
    }
  }
}


