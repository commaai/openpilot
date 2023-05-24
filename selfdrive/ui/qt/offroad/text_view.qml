import QtQuick 2.0

Item {
  id: root
  signal scroll()

  Flickable {
    id: flickArea
    objectName: "flickArea"
    anchors.fill: parent
    contentHeight: helpText.height
    contentWidth: width - (leftMargin + rightMargin)
    bottomMargin: 50
    topMargin: 50
    rightMargin: 50
    leftMargin: 50
    flickableDirection: Flickable.VerticalFlick
    flickDeceleration: 7500.0
    maximumFlickVelocity: 10000.0
    pixelAligned: true

    onAtYEndChanged: root.scroll()

    Text {
      id: helpText
      width: flickArea.contentWidth
      font.family: "Inter"
      font.weight: "Light"
      font.pixelSize: 50
      textFormat: Text.RichText
      color: "#C9C9C9"
      wrapMode: Text.Wrap
      text: text_view
    }
  }

  Rectangle {
    id: scrollbar
    anchors.right: flickArea.right
    anchors.rightMargin: 20
    y: flickArea.topMargin + flickArea.visibleArea.yPosition * (flickArea.height - flickArea.bottomMargin - flickArea.topMargin)
    width: 12
    radius: 6
    height: flickArea.visibleArea.heightRatio * (flickArea.height - flickArea.bottomMargin - flickArea.topMargin)
    color: "#808080"
  }
}
