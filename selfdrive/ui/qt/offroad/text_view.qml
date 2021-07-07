import QtQuick 2.0

Item {
  id: root
  signal qmlSignal()

  Flickable {
    id: flickArea
    objectName: "flickArea"
    anchors.fill: parent
    bottomMargin: 50
    topMargin: 50
    leftMargin: 50
    rightMargin: 50
    contentHeight: helpText.height
    contentWidth: helpText.width
    flickableDirection: Flickable.VerticalFlick
    flickDeceleration: 7500.0
    maximumFlickVelocity: 10000.0
    pixelAligned: true

    onAtYEndChanged: root.qmlSignal()

    Text {
      id: helpText
      width: flickArea.width
      font.family: Inter
      font.weight: Light
      font.pixelSize: 50
      textFormat: Text.RichText
      color: "white"
      wrapMode: Text.Wrap
      text: text_view
    }
  }
}


