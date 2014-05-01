namespace Multiplicity
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
            this.backgroundbox = new System.Windows.Forms.PictureBox();
            this.compiledbox = new System.Windows.Forms.PictureBox();
            this.flowLayoutPanel1 = new System.Windows.Forms.FlowLayoutPanel();
            ((System.ComponentModel.ISupportInitialize)(this.backgroundbox)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.compiledbox)).BeginInit();
            this.SuspendLayout();
            // 
            // backgroundbox
            // 
            this.backgroundbox.Location = new System.Drawing.Point(13, 514);
            this.backgroundbox.Name = "backgroundbox";
            this.backgroundbox.Size = new System.Drawing.Size(541, 402);
            this.backgroundbox.TabIndex = 12;
            this.backgroundbox.TabStop = false;
            // 
            // compiledbox
            // 
            this.compiledbox.Location = new System.Drawing.Point(560, 514);
            this.compiledbox.Name = "compiledbox";
            this.compiledbox.Size = new System.Drawing.Size(541, 402);
            this.compiledbox.TabIndex = 13;
            this.compiledbox.TabStop = false;
            // 
            // flowLayoutPanel1
            // 
            this.flowLayoutPanel1.AutoScroll = true;
            this.flowLayoutPanel1.Location = new System.Drawing.Point(62, 12);
            this.flowLayoutPanel1.Name = "flowLayoutPanel1";
            this.flowLayoutPanel1.Size = new System.Drawing.Size(965, 496);
            this.flowLayoutPanel1.TabIndex = 14;
            this.flowLayoutPanel1.Paint += new System.Windows.Forms.PaintEventHandler(this.flowLayoutPanel1_Paint);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1121, 1025);
            this.Controls.Add(this.flowLayoutPanel1);
            this.Controls.Add(this.compiledbox);
            this.Controls.Add(this.backgroundbox);
            this.Margin = new System.Windows.Forms.Padding(4);
            this.Name = "Form1";
            this.Text = "Multiplicity";
            ((System.ComponentModel.ISupportInitialize)(this.backgroundbox)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.compiledbox)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.ToolTip toolTip1;
        private System.Windows.Forms.PictureBox backgroundbox;
        private System.Windows.Forms.PictureBox compiledbox;
        private System.Windows.Forms.FlowLayoutPanel flowLayoutPanel1;
    }
}

